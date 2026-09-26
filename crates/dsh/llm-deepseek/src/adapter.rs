//! The DeepSeek adapter, ported from
//! `packages/llm/llm-deepseek/src/adapter.ts`: reqwest + SSE against an
//! OpenAI-compatible chat-completions endpoint, emitting harness stream
//! chunks. Transport-only — connection facts arrive through a thunk resolved
//! once per operation, and the bearer token through a per-request resolver
//! taking that same snapshot, so an endpoint and the secret sent to it can
//! never come from different configuration generations.
//!
//! Divergences:
//! - The idle watchdog is a per-read `tokio::time::timeout` (any transport
//!   read, including comment keep-alives, resets it) instead of upstream's
//!   pulse-driven watchdog — the same observable contract.
//! - `resolve_user_id` returns a plain string until the identity crate's type
//!   is wired through.

use crate::serialize::{RequestDefaults, serialize_request};
use crate::sse::{DONE, SseDecoder};
use crate::translate::Translator;
use crate::wire::WireError;
use async_stream::stream;
use dsh_llm::{
    AdapterStream, CONTEXT_WINDOW_EXCEEDED_CODE, GenerateOptions, LlmAdapter, LlmError,
    LlmModelContext, LlmModelInfo, LlmModelReasoningInfo, LlmProviderInfo, LlmReasoningEffortInfo,
    LlmResolvedModelInfo, ModelModality, ProviderRequestId, QUOTA_EXCEEDED_CODE, ReasoningEffortId,
    ResolvedRetryPolicy, attribution_headers, is_context_window_exceeded_error,
    is_quota_exceeded_error,
};
use futures::future::LocalBoxFuture;
use futures::{FutureExt, StreamExt};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Duration;

/// Default maximum idle interval while a stream read is outstanding.
pub const DEFAULT_STREAM_IDLE_TIMEOUT_MS: u64 = 300_000;
/// Default combined request/response context capacity.
pub const DEFAULT_CONTEXT_WINDOW: u64 = 1_000_000;
/// Default per-request output-token cap.
pub const DEFAULT_MAX_TOKENS: u64 = 256_000;

/// One optional model entry advertised by the adapter (upstream
/// `DeepSeekCatalogModel`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeepSeekCatalogModel {
    /// Wire model id accepted by the configured endpoint.
    pub id: String,
    /// Selector label; defaults to the id.
    pub name: Option<String>,
    pub description: Option<String>,
    /// Known context capacity; `None` falls back to the profile default.
    pub context_window: Option<u64>,
    /// Per-model output cap; `None` falls back to the profile default.
    pub max_tokens: Option<u64>,
}

/// Validated connection facts for one operation (upstream
/// `DeepSeekConnectionOptions`). The registering plugin owns validation and
/// layering; the adapter trusts this and re-reads it per operation, which is
/// what makes a configuration change reach the next request without
/// re-registration.
#[derive(Debug, Clone)]
pub struct DeepSeekConnectionOptions {
    /// Endpoint base; `/chat/completions` is appended.
    pub base_url: String,
    /// Credential reference name resolved per request, travelling with the
    /// endpoint so URL and secret share one resolution.
    pub api_key_env: String,
    /// Request defaults applied to every call (thinking mode, effort).
    pub defaults: RequestDefaults,
    /// Default per-request output cap; explicit request values win.
    pub max_tokens: u64,
    /// Context capacity used when the selected model has no exact value.
    pub default_context_window: u64,
    /// Advisory models for discovery; requests remain unrestricted.
    pub models: Vec<DeepSeekCatalogModel>,
    /// Maximum provider idle time while one stream read is outstanding.
    pub stream_idle_timeout_ms: u64,
    /// Provider-owned retry policy, already resolved.
    pub retry_policy: ResolvedRetryPolicy,
    /// Extra HTTP headers for this connection snapshot. 0G trust and sort
    /// ride here; an empty list leaves the request unchanged.
    pub extra_headers: Vec<(String, String)>,
}

/// Per-session 0G dispatch facts. Present only for a routed 0G call.
#[derive(Debug, Clone)]
pub struct RequestOverlay {
    pub base_url: String,
    pub api_key_env: String,
    pub headers: Vec<(String, String)>,
    pub verify_tee: bool,
}

/// Operation-local resolution hooks the registering plugin owns (upstream
/// `DeepSeekAdapterOptions`).
pub struct DeepSeekAdapterOptions {
    /// Current validated connection facts; called once per operation.
    pub options: Box<dyn Fn() -> DeepSeekConnectionOptions>,
    /// Resolve the bearer token for one request's connection snapshot; fails
    /// with `MISSING_CREDENTIAL` when no key is available anywhere.
    pub resolve_api_key: Box<
        dyn Fn(&DeepSeekConnectionOptions) -> LocalBoxFuture<'static, Result<String, LlmError>>,
    >,
    /// Resolve the harness-home anonymous id shared with telemetry.
    pub resolve_user_id: Box<dyn Fn() -> String>,
}

fn model_info(provider: &str, model: &DeepSeekCatalogModel) -> LlmModelInfo {
    LlmModelInfo {
        provider: provider.to_string(),
        id: model.id.clone(),
        name: model.name.clone().unwrap_or_else(|| model.id.clone()),
        description: model.description.clone(),
        input_modalities: Some(vec![ModelModality::Text]),
    }
}

fn efforts(off_only: bool) -> Vec<LlmReasoningEffortInfo> {
    let mut list = vec![LlmReasoningEffortInfo {
        id: ReasoningEffortId::new("off"),
        name: "Off".into(),
        description: None,
    }];
    if !off_only {
        list.push(LlmReasoningEffortInfo {
            id: ReasoningEffortId::new("high"),
            name: "High".into(),
            description: None,
        });
        list.push(LlmReasoningEffortInfo {
            id: ReasoningEffortId::new("max"),
            name: "Max".into(),
            description: None,
        });
    }
    list
}

/// Parse a `retry-after` header value (delta-seconds or HTTP date) to
/// positive milliseconds.
fn provider_retry_after_ms(value: Option<&str>) -> Option<f64> {
    let value = value?;
    if value.chars().all(|c| c.is_ascii_digit()) && !value.is_empty() {
        let delay = value.parse::<f64>().ok()? * 1_000.0;
        return (delay.is_finite() && delay > 0.0).then_some(delay);
    }
    let date = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    let delay = (date.timestamp_millis() - chrono::Utc::now().timestamp_millis()) as f64;
    (delay > 0.0).then_some(delay)
}

fn request_id(headers: &reqwest::header::HeaderMap) -> Option<ProviderRequestId> {
    let value = headers
        .get("x-request-id")
        .or_else(|| headers.get("x-deepseek-request-id"))?
        .to_str()
        .ok()?;
    (!value.is_empty()).then(|| ProviderRequestId::new(value))
}

/// Map an HTTP status (plus the parsed provider error body) to a stable
/// harness error code (upstream `httpErrorCode`).
pub fn http_error_code(status: u16, error: Option<&crate::wire::WireErrorDetail>) -> String {
    if status == 401 || status == 403 {
        return "AUTH".into();
    }
    let detail: String = error
        .map(|error| {
            [
                error.code.as_deref(),
                error.error_type.as_deref(),
                error.message.as_deref(),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<&str>>()
            .join(" ")
        })
        .unwrap_or_default();
    if is_quota_exceeded_error(&detail) {
        return QUOTA_EXCEEDED_CODE.into();
    }
    if status == 429 {
        return "RATE_LIMIT".into();
    }
    if status == 400 {
        if is_context_window_exceeded_error(&detail) {
            return CONTEXT_WINDOW_EXCEEDED_CODE.into();
        }
        return "INVALID_REQUEST".into();
    }
    if status >= 500 {
        return "SERVER".into();
    }
    format!("HTTP_{status}")
}

/// The direct-fetch DeepSeek adapter. One instance serves every model name it
/// was registered under (the harness model name IS the wire model name).
pub struct DeepSeekAdapter {
    config: Rc<DeepSeekAdapterOptions>,
    client: reqwest::Client,
    overlays: Rc<RefCell<HashMap<String, RequestOverlay>>>,
}

impl DeepSeekAdapter {
    pub fn new(config: DeepSeekAdapterOptions) -> DeepSeekAdapter {
        DeepSeekAdapter {
            config: Rc::new(config),
            client: reqwest::Client::new(),
            overlays: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    /// Session-scoped 0G headers. The bridge writes one entry before the turn
    /// and removes it when the run finishes.
    pub fn overlays(&self) -> Rc<RefCell<HashMap<String, RequestOverlay>>> {
        self.overlays.clone()
    }
}

impl LlmAdapter for DeepSeekAdapter {
    fn provider_info(&self, provider: &str) -> LlmProviderInfo {
        LlmProviderInfo {
            id: provider.to_string(),
            name: "DeepSeek".to_string(),
        }
    }

    fn provider_retry_policy(&self, _provider: &str) -> Option<ResolvedRetryPolicy> {
        Some((self.config.options)().retry_policy)
    }

    fn list_models(&self, provider: &str) -> LocalBoxFuture<'_, anyhow::Result<Vec<LlmModelInfo>>> {
        let connection = (self.config.options)();
        let provider = provider.to_string();
        async move {
            Ok(connection
                .models
                .iter()
                .map(|model| model_info(&provider, model))
                .collect())
        }
        .boxed_local()
    }

    fn resolve_model<'a>(
        &'a self,
        provider: &'a str,
        model: &'a str,
        _signal: Option<dsh_timeout::AbortSignal>,
    ) -> LocalBoxFuture<'a, anyhow::Result<LlmResolvedModelInfo>> {
        let connection = (self.config.options)();
        async move {
            let configured = connection.models.iter().find(|entry| entry.id == model);
            let context_window = configured
                .and_then(|entry| entry.context_window)
                .unwrap_or(connection.default_context_window);
            let base = match configured {
                // Uncatalogued models still declare text-only: this wire route
                // cannot carry images, and "unknown" would let the host accept
                // and persist images the serializer must then reject.
                None => LlmModelInfo {
                    provider: provider.to_string(),
                    id: model.to_string(),
                    name: model.to_string(),
                    description: None,
                    input_modalities: Some(vec![ModelModality::Text]),
                },
                Some(entry) => model_info(provider, entry),
            };
            let thinking_disabled = connection.defaults.thinking.as_deref() == Some("disabled");
            let default_effort = if thinking_disabled {
                "off"
            } else {
                match connection.defaults.reasoning_effort.as_deref() {
                    Some("off") => "off",
                    Some("max") => "max",
                    _ => "high",
                }
            };
            Ok(LlmResolvedModelInfo {
                provider: base.provider,
                id: base.id,
                name: base.name,
                description: base.description,
                input_modalities: base.input_modalities,
                context: Some(LlmModelContext { context_window }),
                default_max_tokens: Some(
                    configured
                        .and_then(|entry| entry.max_tokens)
                        .unwrap_or(connection.max_tokens),
                ),
                reasoning: Some(LlmModelReasoningInfo {
                    efforts: efforts(thinking_disabled),
                    default_effort: Some(ReasoningEffortId::new(default_effort)),
                }),
            })
        }
        .boxed_local()
    }

    fn stream(&self, options: GenerateOptions) -> AdapterStream {
        // One resolution per stream call: connection facts and credential
        // freeze here for the whole request; the next call re-resolves.
        let config = self.config.clone();
        let client = self.client.clone();
        let overlays = self.overlays.clone();
        stream! {
            let mut connection = (config.options)();
            let session_key = options
                .session_id
                .as_ref()
                .map(|id| id.as_str().to_string());
            let overlay = session_key
                .as_ref()
                .and_then(|id| overlays.borrow().get(id).cloned());
            if let Some(overlay) = &overlay {
                connection.base_url = overlay.base_url.clone();
                connection.api_key_env = overlay.api_key_env.clone();
                connection.extra_headers = overlay.headers.clone();
            }
            let api_key = match (config.resolve_api_key)(&connection).await {
                Ok(key) => key,
                Err(error) => {
                    yield Err(anyhow::Error::new(error));
                    return;
                }
            };
            let user_id = (config.resolve_user_id)();

            let mut body = match serialize_request(&options, &connection.defaults) {
                Ok(body) => body,
                Err(error) => {
                    yield Err(anyhow::Error::new(error));
                    return;
                }
            };
            if overlay.as_ref().is_some_and(|item| item.verify_tee) {
                body.verify_tee = Some(true);
            }
            let mut request = client
                .post(format!("{}/chat/completions", connection.base_url))
                .header("authorization", format!("Bearer {api_key}"))
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .header("x-deepseek-harness-user-id", user_id)
                .json(&body);
            for (name, value) in attribution_headers(None) {
                request = request.header(name, value);
            }
            if let Some(session_id) = &options.session_id {
                request = request.header("x-deepseek-harness-session-id", session_id.as_str());
            }
            if options.purpose == Some(dsh_llm::CallPurpose::Compaction) {
                request = request.header("x-deepseek-harness-compact", "1");
            }
            for (name, value) in &connection.extra_headers {
                request = request.header(name, value);
            }

            let aborted = || options.signal.as_ref().map(|s| s.aborted()).unwrap_or(false);
            let response = match request.send().await {
                Ok(response) => response,
                Err(error) => {
                    yield Err(if aborted() {
                        anyhow::Error::new(LlmError::new("DeepSeek request aborted by caller", "ABORTED"))
                    } else {
                        // Transport failures (DNS, refused connection, TLS,
                        // proxy) get the endpoint in the message so the chain
                        // renders a full diagnosis.
                        anyhow::Error::new(error).context(LlmError::new(
                            format!("DeepSeek API request to {} failed", connection.base_url),
                            "TRANSPORT",
                        ))
                    });
                    return;
                }
            };

            if overlay.is_some() {
                for (name, value) in response.headers().iter() {
                    let header = name.as_str();
                    if header.contains("0g") || header.starts_with("zg-") {
                        tracing::info!(
                            header,
                            value = value.to_str().unwrap_or(""),
                            "0G response header"
                        );
                    }
                }
            }

            let status = response.status().as_u16();
            if !(200..300).contains(&status) {
                let delay = provider_retry_after_ms(
                    response.headers().get("retry-after").and_then(|v| v.to_str().ok()),
                );
                let id = request_id(response.headers());
                let parsed: Option<WireError> = response.json().await.ok();
                let provider_error = parsed.as_ref().and_then(|body| body.error.as_ref());
                let message = provider_error
                    .and_then(|error| error.message.clone())
                    .unwrap_or_else(|| format!("DeepSeek API error (HTTP {status})"));
                let code = http_error_code(status, provider_error);
                yield Err(anyhow::Error::new(
                    LlmError::new(message, code).with_facts(Some(status), delay, id),
                ));
                return;
            }

            // Stream the SSE body: idle timeout per read (any transport
            // activity, comments included, resets it).
            let idle = Duration::from_millis(connection.stream_idle_timeout_ms.max(1));
            let mut bytes = response.bytes_stream();
            let mut decoder = SseDecoder::new();
            let mut translator = Translator::new();
            let mut finished = false;
            'transport: loop {
                if aborted() {
                    yield Err(anyhow::Error::new(LlmError::new(
                        "DeepSeek request aborted by caller",
                        "ABORTED",
                    )));
                    return;
                }
                let read = tokio::time::timeout(idle, bytes.next()).await;
                let item = match read {
                    Err(_) => {
                        yield Err(anyhow::Error::new(LlmError::new(
                            format!(
                                "DeepSeek stream idle timeout after {}ms",
                                connection.stream_idle_timeout_ms
                            ),
                            "TIMEOUT",
                        )));
                        return;
                    }
                    Ok(None) => break 'transport,
                    Ok(Some(Err(error))) => {
                        yield Err(if aborted() {
                            anyhow::Error::new(LlmError::new(
                                "DeepSeek request aborted by caller",
                                "ABORTED",
                            ))
                        } else {
                            anyhow::Error::new(error).context(LlmError::new(
                                format!("DeepSeek API stream from {} failed", connection.base_url),
                                "TRANSPORT",
                            ))
                        });
                        return;
                    }
                    Ok(Some(Ok(item))) => item,
                };
                for payload in decoder.feed(&item, |_comment| {}) {
                    if payload == DONE {
                        for chunk in translator.done() {
                            yield Ok(chunk);
                        }
                        finished = true;
                        break 'transport;
                    }
                    match translator.feed(&payload) {
                        Ok(chunks) => {
                            for chunk in chunks {
                                yield Ok(chunk);
                            }
                        }
                        Err(error) => {
                            yield Err(anyhow::Error::new(error));
                            return;
                        }
                    }
                }
            }
            if !finished {
                // 0G closes a finished completion without the OpenAI `[DONE]`
                // sentinel. DeepSeek's own API still treats that EOF as
                // truncation.
                if og_endpoint(&connection) && translator.has_terminal_progress() {
                    for chunk in translator.done() {
                        yield Ok(chunk);
                    }
                } else {
                    yield Err(anyhow::Error::new(LlmError::new(
                        "SSE stream ended without [DONE]",
                        "STREAM_CLOSED",
                    )));
                }
            }
        }
        .boxed_local()
    }
}

/// 0G Router speaks chat-completions but ends a finished stream by closing
/// the body, without `data: [DONE]`.
fn og_endpoint(connection: &DeepSeekConnectionOptions) -> bool {
    connection.api_key_env == "OG_API_KEY" || connection.base_url.contains("0g.ai")
}

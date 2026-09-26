//! In-process bridge exposing the dsh runtime as a comet/keel [`Harness`].
//!
//! The dsh runtime is single-threaded by design (Rc + tokio LocalSet,
//! mirroring the upstream JS event loop), while comet's `Harness` trait is
//! `Send + Sync` with `'static` Send streams. The bridge is therefore an
//! actor: one dedicated OS thread runs the dsh composition on its own
//! current-thread runtime, and [`DshHarness`] is a thin Send handle that
//! passes commands over a channel and hands the engine a Send event stream.
//!
//! Event translation: dsh session events (the durable log feed) map onto
//! comet `AgentEvent`s — text/reasoning deltas stream as they land in the
//! log, tool call/result pairs map by call id, usage rides through, and the
//! closing `turn/end` decides `Done { status }`. Sessions persist as normal
//! dsh JSONL artifacts under `$DSH_HOME/sessions`, and `RunRequest::resume`
//! carries the dsh session id back in, so keel's resume flow drives dsh
//! session continuity natively.
//!
//! Deferred: image attachments (the DeepSeek chat-completions route is
//! text-only), reasoning-level mapping, and mid-run input requests (dsh's
//! ask-user tier is not ported yet).

use async_trait::async_trait;
use dsh_agent::{AgentOptions, AgentRegistry, CreateAgentOptions, ResumeAgentOptions};
use dsh_agent_loop::{AgentLoop, DecisionActivityChanged};
use dsh_cordis::{App, Context};
use dsh_llm::{
    ContentBlock, LlmRuntime, MessageSource, StreamChunk, assert_usable_api_key,
    create_user_message,
};
use dsh_llm_deepseek::{
    DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS, DEFAULT_STREAM_IDLE_TIMEOUT_MS, DeepSeekAdapter,
    DeepSeekAdapterOptions, DeepSeekConnectionOptions, RequestDefaults, RequestOverlay,
};
use dsh_session::{SessionEventData, SessionId, SessionMeta, SessionStore, TurnEndReason};
use dsh_session_persistence::{PersistenceService, install_write_behind};
use dsh_session_persistence_jsonl::JsonlPersistence;
use dsh_system_prompt::{Config as PromptConfig, SystemPrompt};
use dsh_tools::{Config as ToolsConfig, ToolRuntime};
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use keel_harness::{Harness, HarnessError, RunControls, SteerMessage};
use keel_proto::{
    AgentEvent, DecisionEvent, DoneStatus, HarnessId, Model, ReasoningLevel, RunRequest,
    SteeringMode, ToolCall,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Mutex;
use tokio::sync::mpsc;

pub mod og;

/// Event sender type: futures channel so the receiver is a Send `Stream`.
type EventTx = futures::channel::mpsc::UnboundedSender<Result<AgentEvent, HarnessError>>;

enum BridgeCommand {
    Run {
        request: RunRequest,
        events: EventTx,
        steering: mpsc::Receiver<SteerMessage>,
        interrupt: tokio_util::sync::CancellationToken,
    },
}

/// The comet-facing harness handle. Construction spawns the dsh actor thread
/// lazily on first use (the registry's lazy-slot factory constructs this, so
/// listing harnesses never boots the runtime).
pub struct DshHarness {
    commands: Mutex<Option<mpsc::UnboundedSender<BridgeCommand>>>,
    decision_data_dir: Option<PathBuf>,
    kind: LoopKind,
}

/// Which in-process loop this actor serves. DeepSeek and 0G share the
/// OpenAI-compatible tool loop, but each keeps its own endpoint, credential,
/// and catalog.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LoopKind {
    Deepseek,
    Og,
}

/// The embedded DeepSeek adapter needs `DEEPSEEK_API_KEY`. Laya and Jev
/// choose bounded actions but do not supply a generation credential.
pub fn deepseek_credential_available() -> bool {
    std::env::var("DEEPSEEK_API_KEY").ok().is_some_and(|raw| {
        assert_usable_api_key(&raw, "dsh-llm-deepseek", "DEEPSEEK_API_KEY").is_ok()
    })
}

impl Default for DshHarness {
    fn default() -> Self {
        Self::new()
    }
}

impl DshHarness {
    pub fn new() -> DshHarness {
        Self::with_kind(LoopKind::Deepseek, None)
    }

    /// Reads the local decision preference from the app data directory.
    pub fn with_decision_data_dir(path: PathBuf) -> DshHarness {
        Self::with_kind(LoopKind::Deepseek, Some(path))
    }

    /// 0G Router loop. Does not replace [`Self::new`]; DeepSeek stays on the
    /// direct API even when `OG_API_KEY` is set.
    pub fn og() -> DshHarness {
        Self::with_kind(LoopKind::Og, None)
    }

    /// 0G Router loop with the local decision preference directory.
    pub fn og_with_decision_data_dir(path: PathBuf) -> DshHarness {
        Self::with_kind(LoopKind::Og, Some(path))
    }

    fn with_kind(kind: LoopKind, decision_data_dir: Option<PathBuf>) -> DshHarness {
        DshHarness {
            commands: Mutex::new(None),
            decision_data_dir,
            kind,
        }
    }

    /// The command channel, booting the actor thread on first use.
    fn commands(&self) -> mpsc::UnboundedSender<BridgeCommand> {
        let mut slot = self.commands.lock().expect("bridge command slot");
        if let Some(sender) = &*slot {
            if !sender.is_closed() {
                return sender.clone();
            }
        }
        let (tx, rx) = mpsc::unbounded_channel();
        let decision_data_dir = self.decision_data_dir.clone();
        let kind = self.kind;
        std::thread::Builder::new()
            .name(
                match kind {
                    LoopKind::Og => "og-router",
                    LoopKind::Deepseek => "dsh-harness",
                }
                .into(),
            )
            .spawn(move || actor_thread(rx, decision_data_dir, kind))
            .expect("spawn in-process harness thread");
        *slot = Some(tx.clone());
        tx
    }
}

#[async_trait]
impl Harness for DshHarness {
    fn id(&self) -> HarnessId {
        match self.kind {
            LoopKind::Deepseek => HarnessId::Dsh,
            LoopKind::Og => HarnessId::Og,
        }
    }

    fn display_name(&self) -> &str {
        match self.kind {
            LoopKind::Deepseek => "DeepSeek Harness",
            LoopKind::Og => "0G Router",
        }
    }

    fn supports_steering(&self) -> bool {
        true
    }

    fn steering_mode(&self) -> SteeringMode {
        SteeringMode::StepBoundary
    }

    fn reasoning_levels(&self) -> &[ReasoningLevel] {
        &[ReasoningLevel::Medium]
    }

    /// Each loop is selectable only when its own credential is present.
    fn installed(&self) -> bool {
        match self.kind {
            LoopKind::Deepseek => deepseek_credential_available(),
            LoopKind::Og => og::api_key_available(),
        }
    }

    async fn models(&self) -> Result<Vec<Model>, HarnessError> {
        if self.kind == LoopKind::Og {
            return Ok(og::load_catalog()
                .await
                .iter()
                .map(og::to_harness_model)
                .collect());
        }
        Ok(vec![
            Model {
                id: "deepseek-chat".into(),
                label: "DeepSeek Chat".into(),
                description: None,
                reasoning_levels: vec![ReasoningLevel::Medium],
                options: vec![],
            },
            Model {
                id: "deepseek-reasoner".into(),
                label: "DeepSeek Reasoner".into(),
                description: None,
                reasoning_levels: vec![ReasoningLevel::Medium],
                options: vec![],
            },
        ])
    }

    async fn run(
        &self,
        request: RunRequest,
        controls: RunControls,
    ) -> Result<BoxStream<'static, Result<AgentEvent, HarnessError>>, HarnessError> {
        let (events_tx, events_rx) = futures::channel::mpsc::unbounded();
        self.commands()
            .send(BridgeCommand::Run {
                request,
                events: events_tx,
                steering: controls.steering,
                interrupt: controls.interrupt,
            })
            .map_err(|_| HarnessError::Protocol("dsh bridge thread is gone".into()))?;
        Ok(events_rx.boxed())
    }
}

/// Flatten a message's text blocks (tool-result output projection), capped.
fn flatten_text_capped(content: &[ContentBlock], cap: usize) -> Option<String> {
    let text: String = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if text.is_empty() {
        return None;
    }
    if text.len() <= cap {
        return Some(text);
    }
    let mut capped: String = text.chars().take(cap).collect();
    capped.push('…');
    Some(capped)
}

/// Map a dsh tool invocation onto comet's decoded tool-call vocabulary; names
/// outside the known set render through `Unknown` (generic card).
fn map_tool_call(name: &str, raw_arguments: &str) -> ToolCall {
    let args: serde_json::Value = serde_json::from_str(raw_arguments).unwrap_or_default();
    let string_arg = |key: &str| args.get(key).and_then(|v| v.as_str()).map(str::to_string);
    let file_path_arg = || {
        string_arg("file_path")
            .or_else(|| string_arg("path"))
            .unwrap_or_default()
    };
    match name {
        "bash" | "shell" => ToolCall::Exec {
            command: string_arg("command").unwrap_or_else(|| raw_arguments.to_string()),
        },
        // dsh's fs tools are named `read`/`write`/`edit` and take
        // `file_path` (upstream `packages/fs/tool-fs`); `path` and the
        // `*_file` spellings are accepted for out-of-tree tools using them.
        "read" | "read_file" => ToolCall::ReadFile {
            path: file_path_arg(),
        },
        "write" | "write_file" => ToolCall::WriteFile {
            path: file_path_arg(),
            content: string_arg("content"),
        },
        "edit" | "edit_file" | "str_replace_editor" => ToolCall::EditFile {
            path: file_path_arg(),
            old_string: string_arg("old_string"),
            new_string: string_arg("new_string"),
        },
        "grep" => ToolCall::Search {
            pattern: string_arg("pattern").unwrap_or_default(),
            path: string_arg("path"),
        },
        "glob" => ToolCall::Glob {
            pattern: string_arg("pattern").unwrap_or_default(),
        },
        other => ToolCall::Unknown {
            name: other.to_string(),
            input: (!args.is_null()).then_some(args),
        },
    }
}

#[cfg(test)]
mod tool_mapping_tests {
    use super::map_tool_call;
    use keel_proto::ToolCall;

    /// The names here are the tools dsh actually registers (upstream
    /// `packages/fs/tool-fs`, `tool-fs-search`, `shell/tool-shell`); a rename
    /// on either side must fail here rather than silently degrade every card
    /// in the transcript to the generic renderer.
    #[test]
    fn dsh_tool_names_map_to_typed_cards() {
        assert!(matches!(
            map_tool_call("bash", r#"{"command":"ls -la"}"#),
            ToolCall::Exec { command } if command == "ls -la"
        ));
        // dsh's fs tools take `file_path` (upstream `parseReadArgs`), so the
        // card must read that key — reading `path` here would render every
        // file card blank.
        assert!(matches!(
            map_tool_call("read", r#"{"file_path":"/tmp/a.rs"}"#),
            ToolCall::ReadFile { path } if path == "/tmp/a.rs"
        ));
        assert!(matches!(
            map_tool_call("write", r#"{"file_path":"/tmp/a.rs","content":"x"}"#),
            ToolCall::WriteFile { path, content } if path == "/tmp/a.rs" && content.as_deref() == Some("x")
        ));
        assert!(matches!(
            map_tool_call("edit", r#"{"file_path":"/tmp/a.rs","old_string":"a","new_string":"b"}"#),
            ToolCall::EditFile { path, old_string, new_string }
                if path == "/tmp/a.rs" && old_string.as_deref() == Some("a")
                    && new_string.as_deref() == Some("b")
        ));
        // The `path` spelling stays accepted for out-of-tree tools.
        assert!(matches!(
            map_tool_call("read_file", r#"{"path":"/tmp/b.rs"}"#),
            ToolCall::ReadFile { path } if path == "/tmp/b.rs"
        ));
        assert!(matches!(
            map_tool_call("glob", r#"{"pattern":"**/*.rs"}"#),
            ToolCall::Glob { pattern } if pattern == "**/*.rs"
        ));
        assert!(matches!(
            map_tool_call("grep", r#"{"pattern":"fn main","path":"src"}"#),
            ToolCall::Search { pattern, path }
                if pattern == "fn main" && path.as_deref() == Some("src")
        ));
    }

    #[test]
    fn unknown_names_and_malformed_arguments_degrade_safely() {
        assert!(matches!(
            map_tool_call("some_plugin_tool", r#"{"a":1}"#),
            ToolCall::Unknown { name, .. } if name == "some_plugin_tool"
        ));
        // A model can emit invalid JSON; the card must still render.
        assert!(matches!(
            map_tool_call("bash", "not json"),
            ToolCall::Exec { command } if command == "not json"
        ));
    }
}

/// Translate one committed dsh session event into zero or more AgentEvents.
fn translate(event: &dsh_session::SessionEvent) -> Vec<AgentEvent> {
    match &event.data {
        SessionEventData::Extension { event_type, data } if event_type == "decision/receipt" => {
            serde_json::from_value::<DecisionEvent>(data.clone())
                .map(|event| {
                    vec![AgentEvent::Decision {
                        event: event.bounded(),
                    }]
                })
                .unwrap_or_default()
        }
        SessionEventData::AssistantChunk { chunk, .. } => match chunk {
            StreamChunk::TextDelta { text, .. } => {
                vec![AgentEvent::TextDelta { text: text.clone() }]
            }
            StreamChunk::ReasoningDelta { text, .. } => {
                vec![AgentEvent::ReasoningDelta { text: text.clone() }]
            }
            StreamChunk::Usage { usage } => vec![AgentEvent::Usage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
            }],
            _ => vec![],
        },
        SessionEventData::AssistantMessage { message, .. } => {
            vec![AgentEvent::AssistantMessageCompleted {
                assistant_message_id: message.id.as_str().to_string(),
            }]
        }
        SessionEventData::ToolCall {
            call_id,
            name,
            arguments,
            ..
        } => vec![AgentEvent::ToolCall {
            id: call_id.as_str().to_string(),
            call: map_tool_call(name, arguments),
        }],
        SessionEventData::ToolResult { message, .. } => {
            let (call_id, is_error, output) = match message.content.first() {
                Some(ContentBlock::ToolResult {
                    tool_call_id,
                    is_error,
                    content,
                }) => (
                    tool_call_id.as_str().to_string(),
                    is_error.unwrap_or(false),
                    flatten_text_capped(content, 4_000),
                ),
                _ => return vec![],
            };
            vec![AgentEvent::ToolResult {
                id: call_id,
                is_error,
                output,
                diff: None,
            }]
        }
        SessionEventData::TurnEnd {
            reason: TurnEndReason::Error { error },
            ..
        } => {
            vec![AgentEvent::Error {
                message: format!("{} ({})", error.message, error.code),
            }]
        }
        _ => vec![],
    }
}

fn register_live_activity_forwarder(
    ctx: &Context,
    forwarders: Rc<RefCell<HashMap<String, EventTx>>>,
) -> anyhow::Result<()> {
    ctx.on::<DecisionActivityChanged, _, _>(
        Default::default(),
        move |_ctx, (session_id, activity)| {
            if let Some(tx) = forwarders.borrow().get(session_id) {
                let _ = tx.unbounded_send(Ok(AgentEvent::DecisionActivity {
                    activity: *activity,
                }));
            }
            async { None }
        },
    )
    .map(|_| ())
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[cfg(test)]
mod decision_translation_tests {
    use super::*;
    use keel_proto::{
        DecisionActivity, DecisionBackend, DecisionCandidate, DecisionPhase, DecisionResult,
        DecisionStage, DecisionValidation,
    };

    #[test]
    fn live_focus_activity_crosses_bridge_without_a_session_event() {
        dsh_cordis::run(async {
            let app = App::new();
            let ctx = app.root();
            let forwarders: Rc<RefCell<HashMap<String, EventTx>>> = Rc::default();
            register_live_activity_forwarder(&ctx, forwarders.clone()).unwrap();
            let (tx, mut rx) = futures::channel::mpsc::unbounded();
            forwarders.borrow_mut().insert("dsh-chat".into(), tx);
            let active = Some(DecisionActivity {
                backend: DecisionBackend::Jev,
                phase: DecisionPhase::ChoosingFocus,
            });
            ctx.emit::<DecisionActivityChanged>(&("dsh-chat".into(), active));
            assert!(matches!(
                rx.next().await.unwrap().unwrap(),
                AgentEvent::DecisionActivity { activity } if activity == active
            ));
            ctx.emit::<DecisionActivityChanged>(&("dsh-chat".into(), None));
            assert!(matches!(
                rx.next().await.unwrap().unwrap(),
                AgentEvent::DecisionActivity { activity: None }
            ));
        });
    }

    #[test]
    fn bounded_dsh_receipt_crosses_the_harness_event_boundary() {
        let receipt = DecisionEvent::new(
            "dsh-1-1-pre-step",
            1,
            DecisionBackend::Jev,
            DecisionStage::PreStep,
            vec![DecisionCandidate {
                id: "inspect".into(),
                summary: "Read files".into(),
            }],
            DecisionResult::Selected {
                candidate_id: "inspect".into(),
            },
            DecisionValidation::Accepted,
            10,
        )
        .with_observed_outcome("completed_without_tool_calls");
        let event = dsh_session::SessionEvent {
            seq: 0,
            time: 10,
            data: SessionEventData::Extension {
                event_type: "decision/receipt".into(),
                data: serde_json::to_value(&receipt).unwrap(),
            },
            source_event_seqs: None,
            surface_op: None,
            ignorable: None,
        };
        let restored: dsh_session::SessionEvent =
            serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap();
        assert!(matches!(
            translate(&restored).as_slice(),
            [AgentEvent::Decision { event: forwarded }] if forwarded == &receipt
        ));
    }
}

/// The dsh composition living on the actor thread: the same stack the `dsh`
/// CLI boots, minus stdout wiring.
struct Composition {
    harness: HarnessId,
    sessions: Rc<SessionStore>,
    agents: Rc<AgentRegistry>,
    /// Live per-session event forwarders, keyed by session id string.
    forwarders: Rc<RefCell<HashMap<String, EventTx>>>,
    /// 0G trust and sort headers for the session the adapter is about to call.
    og_overlays: Rc<RefCell<HashMap<String, RequestOverlay>>>,
    _app: App,
}

/// Register the model-facing toolset: filesystem read/write/edit with the
/// read-before-edit observation policy, glob/grep search, and bash. The tool
/// effects live for the composition's lifetime, so the handles are leaked
/// deliberately into the root context's ownership.
fn install_tools(
    ctx: &dsh_cordis::Context,
    tools: &Rc<ToolRuntime>,
    cwd: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    let fs = dsh_fs::LocalFileSystem::provide(
        ctx,
        dsh_fs::LocalFileSystemConfig {
            cwd,
            diff_basis_max_bytes: None,
        },
    )?;
    // Read-before-edit: an edit to a file this session has not read is
    // refused, and a stale observation (changed on disk since the read) is
    // refused too.
    dsh_fs::install_observation_policy(ctx)?;
    dsh_fs::register_fs_tools(ctx, tools, &fs, dsh_fs::FsToolsConfig::default())?;
    dsh_fs::register_search_tools(ctx, tools, dsh_fs::SearchToolsConfig::default())?;
    let subprocess = dsh_shell::LocalSubprocessRuntime::provide(ctx)?;
    let shell =
        dsh_shell::LocalBashExecutor::provide(ctx, subprocess, dsh_shell::BashConfig::default())?;
    dsh_shell::register_bash_tool(ctx, tools, &shell)?;
    Ok(())
}

fn build_composition(
    decision_data_dir: Option<PathBuf>,
    kind: LoopKind,
) -> anyhow::Result<Composition> {
    let app = App::new();
    let ctx = app.root();
    let home = dsh_home_paths::resolve_dsh_home(None);
    let backend = Rc::new(JsonlPersistence::new(home.join("sessions"))?);

    let sessions = SessionStore::provide(&ctx).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let agents = AgentRegistry::provide(&ctx).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let llm = LlmRuntime::provide(&ctx).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let og = kind == LoopKind::Og;
    let system_prompt = SystemPrompt::new(
        &ctx,
        PromptConfig {
            persona: if og {
                "You are an AI agent. The 0G Router chooses the inference provider. Answer the user's task directly and concisely.".into()
            } else {
                "Answer the user's task directly and concisely.".into()
            },
            include_harness_identity: !og,
            ..PromptConfig::default()
        },
    )?;
    let tools = ToolRuntime::provide(&ctx, ToolsConfig::default())?;
    // The run's cwd arrives per RunRequest; the composition-level default is
    // the process cwd (a resolution base, never a containment boundary).
    install_tools(&ctx, &tools, None)?;
    PersistenceService::provide(&ctx, backend.clone())
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    install_write_behind(&ctx, backend, 200).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    {
        let provider = tools.schemas_provider();
        system_prompt.tools(&ctx, move |context| {
            let result = provider(context.scope.as_ref());
            dsh_system_prompt::ToolProviderResult {
                schemas: result.schemas,
                known_names: Some(result.known_names),
            }
        })?;
    }

    let og_endpoint = kind == LoopKind::Og;
    let adapter = Rc::new(DeepSeekAdapter::new(DeepSeekAdapterOptions {
        options: Box::new(move || {
            DeepSeekConnectionOptions {
                base_url: if og_endpoint {
                    og::base_url()
                } else {
                    std::env::var("DEEPSEEK_BASE_URL")
                        .unwrap_or_else(|_| "https://api.deepseek.com".into())
                        .trim_end_matches('/')
                        .to_string()
                },
                api_key_env: if og_endpoint {
                    "OG_API_KEY".into()
                } else {
                    "DEEPSEEK_API_KEY".into()
                },
                defaults: RequestDefaults::default(),
                max_tokens: DEFAULT_MAX_TOKENS,
                default_context_window: DEFAULT_CONTEXT_WINDOW,
                models: vec![],
                stream_idle_timeout_ms: DEFAULT_STREAM_IDLE_TIMEOUT_MS,
                retry_policy: dsh_llm::resolve_retry_policy(None, "dsh: deepseek retryPolicy")
                    .expect("default retry policy resolves"),
                extra_headers: if og_endpoint {
                    vec![(
                        "X-0G-Provider-Trust-Mode".into(),
                        trust_header(og::trust_from_env()),
                    )]
                } else {
                    Vec::new()
                },
            }
        }),
        resolve_api_key: Box::new(|connection| {
            let reference = connection.api_key_env.clone();
            async move {
                let raw = std::env::var(&reference).map_err(|_| {
                    dsh_llm::LlmError::new(
                        format!("no API key: set {reference} in the environment"),
                        "MISSING_CREDENTIAL",
                    )
                })?;
                assert_usable_api_key(&raw, "dsh-llm-deepseek", &reference)
            }
            .boxed_local()
        }),
        resolve_user_id: Box::new(|| "dsh-rs".to_string()),
    }));
    let og_overlays = adapter.overlays();
    llm.register_adapter(&["deepseek".to_string()], adapter)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    AgentLoop::install_with_decision_data_dir(
        &ctx,
        agents.clone(),
        sessions.clone(),
        llm,
        tools,
        system_prompt,
        decision_data_dir,
    )?;

    // One global feed subscription fans committed events out to the live
    // per-session forwarders.
    let forwarders: Rc<RefCell<HashMap<String, EventTx>>> = Rc::default();
    {
        let forwarders = forwarders.clone();
        ctx.on::<dsh_session::SessionEventPublished, _, _>(
            Default::default(),
            move |_ctx, (session, event)| {
                let key = session.id().as_str().to_string();
                if let Some(tx) = forwarders.borrow().get(&key) {
                    for agent_event in translate(event) {
                        let _ = tx.unbounded_send(Ok(agent_event));
                    }
                }
                async { None }
            },
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    }
    // Selector activity is live status, never a dsh session-log event.
    register_live_activity_forwarder(&ctx, forwarders.clone())?;

    Ok(Composition {
        harness: match kind {
            LoopKind::Deepseek => HarnessId::Dsh,
            LoopKind::Og => HarnessId::Og,
        },
        sessions,
        agents,
        forwarders,
        og_overlays,
        _app: app,
    })
}

fn trust_header(trust: og::TrustMode) -> String {
    match trust {
        og::TrustMode::Private => "private".into(),
        og::TrustMode::Verified => "verified".into(),
        og::TrustMode::Standard => "standard".into(),
    }
}

fn overlay_for(request: &RunRequest) -> Option<RequestOverlay> {
    let sort = request
        .model_options
        .get("ogProviderSort")
        .and_then(|value| value.as_str())?;
    let trust = request
        .model_options
        .get("ogTrustMode")
        .and_then(|value| value.as_str())
        .unwrap_or("standard");
    let mut headers = vec![("X-0G-Provider-Trust-Mode".into(), trust.to_string())];
    if sort != "default" {
        headers.push(("X-0G-Provider-Sort".into(), sort.to_string()));
    }
    Some(RequestOverlay {
        base_url: og::base_url(),
        api_key_env: "OG_API_KEY".into(),
        headers,
        verify_tee: request
            .model_options
            .get("ogVerifyTee")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
    })
}

fn actor_thread(
    mut commands: mpsc::UnboundedReceiver<BridgeCommand>,
    decision_data_dir: Option<PathBuf>,
    kind: LoopKind,
) {
    let runtime = match kind {
        LoopKind::Og => "0G router",
        LoopKind::Deepseek => "dsh",
    };
    dsh_cordis::run(async move {
        let composition = match build_composition(decision_data_dir, kind) {
            Ok(composition) => Rc::new(composition),
            Err(error) => {
                tracing::error!("{runtime} bridge composition failed: {error:#}");
                // Drain commands, failing each run loudly.
                while let Some(BridgeCommand::Run { events, .. }) = commands.recv().await {
                    let _ = events.unbounded_send(Ok(AgentEvent::Error {
                        message: format!("{runtime} runtime failed to start: {error:#}"),
                    }));
                    let _ = events.unbounded_send(Ok(AgentEvent::Done {
                        status: DoneStatus::Errored,
                        result: None,
                        error: Some(format!("{error:#}")),
                        session_id: None,
                    }));
                }
                return;
            }
        };
        while let Some(command) = commands.recv().await {
            let BridgeCommand::Run {
                request,
                events,
                steering,
                interrupt,
            } = command;
            let composition = composition.clone();
            tokio::task::spawn_local(async move {
                if let Err(error) =
                    drive_run(&composition, request, events.clone(), steering, interrupt).await
                {
                    let _ = events.unbounded_send(Ok(AgentEvent::Error {
                        message: format!("{error:#}"),
                    }));
                    let _ = events.unbounded_send(Ok(AgentEvent::Done {
                        status: DoneStatus::Errored,
                        result: None,
                        error: Some(format!("{error:#}")),
                        session_id: None,
                    }));
                }
            });
        }
    });
}

/// Drive one keel run: create or resume the dsh agent, forward its session
/// feed, deliver steering and interrupt, and close with `Done`.
async fn drive_run(
    composition: &Rc<Composition>,
    request: RunRequest,
    events: EventTx,
    mut steering: mpsc::Receiver<SteerMessage>,
    interrupt: tokio_util::sync::CancellationToken,
) -> anyhow::Result<()> {
    let model = request.model.clone().unwrap_or_else(|| {
        if composition.harness == HarnessId::Og {
            "glm-5.2".into()
        } else {
            "deepseek-chat".into()
        }
    });
    let agent_options = AgentOptions {
        provider: Some("deepseek".into()),
        model: Some(model.clone()),
        max_tokens: None,
    };
    let handle = match &request.resume {
        Some(id) => {
            composition
                .agents
                .resume(ResumeAgentOptions {
                    resume_session_id: SessionId::new(id.clone()),
                    agent_options,
                })
                .await?
        }
        None => {
            let session_id = SessionId::new(format!("session-{}", uuid::Uuid::new_v4()));
            composition
                .agents
                .create(CreateAgentOptions {
                    session_id,
                    meta: SessionMeta {
                        cwd: (!request.cwd.is_empty()).then(|| request.cwd.clone()),
                        ..Default::default()
                    },
                    seed: vec![],
                    agent_options,
                })
                .await?
        }
    };
    let session_id = handle.agent.id().as_str().to_string();
    // Overlays rewrite the endpoint. Only the 0G loop may apply them, so a
    // DeepSeek run keeps api.deepseek.com and DEEPSEEK_API_KEY.
    if composition.harness == HarnessId::Og
        && let Some(overlay) = overlay_for(&request)
    {
        composition
            .og_overlays
            .borrow_mut()
            .insert(session_id.clone(), overlay);
    }

    let _ = events.unbounded_send(Ok(AgentEvent::SessionStarted {
        harness: composition.harness,
        model,
        tools: vec![],
        cwd: request.cwd.clone(),
        session_id: session_id.clone(),
        assistant_message_id: uuid::Uuid::new_v4().to_string(),
    }));

    // Register the event forwarder before any turn work commits.
    composition
        .forwarders
        .borrow_mut()
        .insert(session_id.clone(), events.clone());

    // Steering and interrupt run beside the turn.
    let steer_agent = handle.agent.clone();
    let steering_task = tokio::task::spawn_local(async move {
        while let Some(steer) = steering.recv().await {
            steer_agent.steer(create_user_message(
                vec![ContentBlock::Text { text: steer.prompt }],
                MessageSource::User,
            ));
        }
    });
    let interrupt_agent = handle.agent.clone();
    let interrupt_token = interrupt.clone();
    let interrupt_task = tokio::task::spawn_local(async move {
        interrupt_token.cancelled().await;
        interrupt_agent.cancel(
            dsh_session::AgentCancelCause::User,
            dsh_agent::CancelOptions::default(),
        );
    });

    handle.agent.followup(create_user_message(
        vec![ContentBlock::Text {
            text: request.prompt.clone(),
        }],
        MessageSource::User,
    ));
    handle.agent.when_idle().await;

    steering_task.abort();
    interrupt_task.abort();

    // Close with the last turn's outcome.
    let outcome = handle.agent.session().with_events(|log| {
        log.iter().rev().find_map(|event| match &event.data {
            SessionEventData::TurnEnd { reason, .. } => Some(reason.clone()),
            _ => None,
        })
    });
    let done = match outcome {
        Some(TurnEndReason::Aborted { .. }) => AgentEvent::Done {
            status: DoneStatus::Interrupted,
            result: None,
            error: None,
            session_id: Some(session_id.clone()),
        },
        Some(TurnEndReason::Error { error }) => AgentEvent::Done {
            status: DoneStatus::Errored,
            result: None,
            error: Some(format!("{} ({})", error.message, error.code)),
            session_id: Some(session_id.clone()),
        },
        _ => AgentEvent::Done {
            status: DoneStatus::Completed,
            result: None,
            error: None,
            session_id: Some(session_id.clone()),
        },
    };

    // Durability, then teardown: the session artifact survives for resume.
    let session = handle.agent.session();
    if let Err(error) = composition.sessions.flush(&session).await {
        tracing::warn!("dsh bridge flush failed: {}", error.0);
    }
    composition.forwarders.borrow_mut().remove(&session_id);
    composition.og_overlays.borrow_mut().remove(&session_id);
    handle.dispose().await;
    let _ = events.unbounded_send(Ok(done));
    Ok(())
}

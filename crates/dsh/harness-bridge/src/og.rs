//! Live 0G model catalog and the host scoring policy.
//!
//! Laya and Jev pick one model id from this snapshot. The host turns the
//! caller's objective into a provider strategy. 0G Router then chooses the
//! provider that serves that model.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use keel_proto::{Model, ReasoningLevel};
use serde::Deserialize;

const DEFAULT_BASE: &str = "https://router-api.0g.ai/v1";
const CACHE_TTL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    Quality,
    Cost,
    Speed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustMode {
    Standard,
    Verified,
    Private,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OgModel {
    pub id: String,
    pub name: String,
    pub context_length: u64,
    pub provider_count: u32,
    pub capabilities: Vec<String>,
    pub verifiability: String,
    pub prompt_price: f64,
    pub completion_price: f64,
}

#[derive(Deserialize)]
struct CatalogBody {
    data: Option<Vec<RawModel>>,
}

#[derive(Deserialize)]
struct RawModel {
    id: String,
    name: Option<String>,
    context_length: Option<u64>,
    provider_count: Option<u32>,
    capabilities: Option<serde_json::Value>,
    supported_parameters: Option<Vec<String>>,
    verifiability: Option<String>,
    pricing: Option<RawPricing>,
}

#[derive(Deserialize)]
struct RawPricing {
    prompt: Option<serde_json::Value>,
    completion: Option<serde_json::Value>,
}

struct Cache {
    at: Instant,
    models: Vec<OgModel>,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

pub fn base_url() -> String {
    std::env::var("OG_BASE_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_BASE.to_string())
        .trim_end_matches('/')
        .to_string()
}

pub fn api_key_available() -> bool {
    std::env::var("OG_API_KEY")
        .ok()
        .is_some_and(|raw| dsh_llm::assert_usable_api_key(&raw, "0g-router", "OG_API_KEY").is_ok())
}

pub fn objective_from_env() -> Objective {
    match std::env::var("KEEL_OG_OBJECTIVE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("cost") => Objective::Cost,
        Some("speed") => Objective::Speed,
        _ => Objective::Quality,
    }
}

pub fn trust_from_env() -> TrustMode {
    match std::env::var("KEEL_OG_TRUST")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("private") => TrustMode::Private,
        Some("verified") => TrustMode::Verified,
        _ => TrustMode::Standard,
    }
}

pub fn provider_strategy(objective: Objective) -> &'static str {
    match objective {
        Objective::Cost => "price",
        Objective::Speed => "latency",
        Objective::Quality => "default",
    }
}

pub fn verify_tee(trust: TrustMode) -> bool {
    trust != TrustMode::Standard
}

pub fn fallback_models() -> Vec<OgModel> {
    vec![
        OgModel {
            id: "glm-5.2".into(),
            name: "GLM-5.2".into(),
            context_length: 1_048_576,
            provider_count: 3,
            capabilities: vec!["reasoning".into(), "tools".into(), "json".into()],
            verifiability: "TeeML".into(),
            prompt_price: 0.00000483,
            completion_price: 0.00001612,
        },
        OgModel {
            id: "qwen-32b".into(),
            name: "Qwen 32B".into(),
            context_length: 131_072,
            provider_count: 4,
            capabilities: vec!["code".into(), "tools".into(), "json".into()],
            verifiability: "TeeTLS".into(),
            prompt_price: 0.0000024,
            completion_price: 0.0000072,
        },
        OgModel {
            id: "deepseek-16b".into(),
            name: "DeepSeek 16B".into(),
            context_length: 65_536,
            provider_count: 5,
            capabilities: vec!["code".into(), "json".into()],
            verifiability: "standard".into(),
            prompt_price: 0.0000011,
            completion_price: 0.0000033,
        },
    ]
}

pub async fn load_catalog() -> Vec<OgModel> {
    if let Some(cached) = fresh_cache() {
        return cached;
    }
    match fetch_catalog().await {
        Ok(models) if !models.is_empty() => {
            store_cache(&models);
            models
        }
        Ok(_) => fallback_models(),
        Err(error) => {
            tracing::info!("0G catalog unavailable, using fallback models: {error}");
            fresh_cache().unwrap_or_else(fallback_models)
        }
    }
}

pub fn eligible(models: &[OgModel], trust: TrustMode) -> Vec<OgModel> {
    let filtered: Vec<OgModel> = models
        .iter()
        .filter(|model| trust_allows(model, trust))
        .cloned()
        .collect();
    if filtered.is_empty() {
        models.to_vec()
    } else {
        filtered
    }
}

pub fn rank(
    prompt: &str,
    models: &[OgModel],
    objective: Objective,
    trust: TrustMode,
) -> Vec<OgModel> {
    let task = infer_task(prompt);
    let mut ranked = eligible(models, trust);
    ranked.sort_by(|left, right| {
        score(right, &task, objective, trust)
            .partial_cmp(&score(left, &task, objective, trust))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

pub fn to_harness_model(model: &OgModel) -> Model {
    Model {
        id: model.id.clone(),
        label: model.name.clone(),
        description: Some(describe(model)),
        reasoning_levels: vec![ReasoningLevel::Medium],
        options: vec![],
    }
}

fn describe(model: &OgModel) -> String {
    format!(
        "capabilities={}; context={}; providers={}; trust={}; promptPrice={}; completionPrice={}",
        if model.capabilities.is_empty() {
            "general".to_string()
        } else {
            model.capabilities.join(",")
        },
        model.context_length,
        model.provider_count,
        model.verifiability,
        model.prompt_price,
        model.completion_price,
    )
}

fn trust_allows(model: &OgModel, trust: TrustMode) -> bool {
    let tier = model.verifiability.to_ascii_lowercase();
    match trust {
        TrustMode::Private => tier == "teeml" || tier == "private",
        TrustMode::Verified => tier != "standard",
        TrustMode::Standard => true,
    }
}

fn infer_task(prompt: &str) -> &'static str {
    let value = prompt.to_ascii_lowercase();
    if contains_any(
        &value,
        &[
            "contract", "solidity", "code", "bug", "漏洞", "代码", "合约", "审计",
        ],
    ) {
        "code-security"
    } else if contains_any(&value, &["extract", "json", "字段", "提取", "结构化"]) {
        "extraction"
    } else if contains_any(&value, &["summar", "摘要", "总结", "归纳"]) {
        "summarization"
    } else if contains_any(&value, &["agent", "tool", "function", "工具", "智能体"]) {
        "agent-tools"
    } else {
        "general-reasoning"
    }
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

fn score(model: &OgModel, task: &str, objective: Objective, trust: TrustMode) -> f64 {
    let text = format!(
        "{} {} {}",
        model.id,
        model.name,
        model.capabilities.join(" ")
    )
    .to_ascii_lowercase();
    let mut score = 50.0;
    if task == "code-security"
        && contains_any(&text, &["code", "reason", "deepseek", "qwen", "glm"])
    {
        score += 25.0;
    }
    if task == "extraction" && contains_any(&text, &["json", "structured", "qwen", "glm"]) {
        score += 20.0;
    }
    if task == "agent-tools" && text.contains("tool") {
        score += 25.0;
    }
    match objective {
        Objective::Quality => score += (model.context_length as f64 / 100_000.0).min(12.0),
        Objective::Cost => score -= (model.prompt_price + model.completion_price) * 500_000.0,
        Objective::Speed => score += f64::from(model.provider_count) * 2.0,
    }
    if trust == TrustMode::Private && model.verifiability.to_ascii_lowercase() != "teeml" {
        score -= 100.0;
    }
    if trust == TrustMode::Verified && model.verifiability.to_ascii_lowercase() == "standard" {
        score -= 60.0;
    }
    score
}

fn fresh_cache() -> Option<Vec<OgModel>> {
    let guard = CACHE.lock().ok()?;
    let cached = guard.as_ref()?;
    (cached.at.elapsed() < CACHE_TTL).then(|| cached.models.clone())
}

fn store_cache(models: &[OgModel]) {
    if let Ok(mut guard) = CACHE.lock() {
        *guard = Some(Cache {
            at: Instant::now(),
            models: models.to_vec(),
        });
    }
}

async fn fetch_catalog() -> Result<Vec<OgModel>, String> {
    let url = format!("{}/models", base_url());
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(6))
        .build()
        .map_err(|error| error.to_string())?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!("0G catalog returned {}", response.status()));
    }
    let body: CatalogBody = response.json().await.map_err(|error| error.to_string())?;
    Ok(body
        .data
        .unwrap_or_default()
        .into_iter()
        .filter(|model| !model.id.is_empty())
        .map(normalize)
        .collect())
}

fn normalize(raw: RawModel) -> OgModel {
    let capabilities = match raw.capabilities {
        Some(serde_json::Value::Array(items)) => items
            .into_iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect(),
        Some(serde_json::Value::Object(map)) => map
            .into_iter()
            .filter(|(_, value)| value.as_bool() == Some(true))
            .map(|(key, _)| key)
            .collect(),
        _ => raw.supported_parameters.unwrap_or_default(),
    };
    let pricing = raw.pricing.unwrap_or(RawPricing {
        prompt: None,
        completion: None,
    });
    OgModel {
        id: raw.id.clone(),
        name: raw.name.unwrap_or(raw.id),
        context_length: raw.context_length.unwrap_or(0),
        provider_count: raw.provider_count.unwrap_or(0),
        capabilities,
        verifiability: raw.verifiability.unwrap_or_else(|| "standard".into()),
        prompt_price: neuron_price(pricing.prompt),
        completion_price: neuron_price(pricing.completion),
    }
}

fn neuron_price(value: Option<serde_json::Value>) -> f64 {
    let number = match value {
        Some(serde_json::Value::String(text)) => text.parse::<f64>().ok(),
        Some(serde_json::Value::Number(number)) => number.as_f64(),
        _ => None,
    };
    match number {
        Some(price) if price > 1.0 => price / 1e18,
        Some(price) => price,
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_prefers_the_cheaper_code_model() {
        let ranked = rank(
            "fix the rust bug",
            &fallback_models(),
            Objective::Cost,
            TrustMode::Standard,
        );
        assert_eq!(ranked[0].id, "deepseek-16b");
        assert_eq!(provider_strategy(Objective::Cost), "price");
    }

    #[test]
    fn private_trust_keeps_teeml_models() {
        let ranked = rank(
            "summarize",
            &fallback_models(),
            Objective::Quality,
            TrustMode::Private,
        );
        assert!(ranked.iter().all(|model| {
            let tier = model.verifiability.to_ascii_lowercase();
            tier == "teeml" || tier == "private"
        }));
        assert!(verify_tee(TrustMode::Private));
        assert!(!verify_tee(TrustMode::Standard));
    }
}

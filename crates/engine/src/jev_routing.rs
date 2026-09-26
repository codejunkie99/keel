//! Typed routing at a clean session boundary. The registry supplies eligible
//! provider/model pairs; local Laya or optional TypeSafe Jev returns only an
//! opaque ID from that set. The coding harness retains its own tool authority.

use std::path::Path;

use jev_core::{Candidate, DecisionBudget, DecisionState, JevConfig, JevSelector, SelectionInput};
use keel_proto::{
    DecisionBackend, DecisionCandidate, DecisionEvent, DecisionResult, DecisionStage,
    DecisionValidation, HarnessId, ReasoningLevel, RunRequest,
};
use laya_local::LayaSelector;

use crate::registry::HarnessRegistry;
use crate::workflow::{self, PreparedAction, TaskState, ValidationContext};

const MAX_ROUTES: usize = 16;
const MAX_LOCAL_ROUTES: usize = 8;

#[derive(Clone, Debug)]
pub struct RouteChoice {
    pub harness: HarnessId,
    pub model: Option<String>,
    /// `default`, `latency`, or `price`. The host sets this; 0G applies it.
    pub provider_strategy: String,
    /// `standard`, `verified`, or `private`.
    pub trust_mode: String,
    pub verify_tee: bool,
}

/// The selected route and its factual decision receipt. A receipt exists only
/// when a selector was actually called; host-only eligibility failures do not
/// appear as model decisions.
pub struct RouteDecision {
    pub choice: Option<RouteChoice>,
    pub event: Option<DecisionEvent>,
}

impl RouteDecision {
    fn skipped() -> Self {
        Self {
            choice: None,
            event: None,
        }
    }
}

struct PreparedRoute {
    id: String,
    choice: RouteChoice,
    description: String,
    provider_name: String,
    model_label: String,
    reasoning_levels: Vec<ReasoningLevel>,
    confidence: f64,
}

pub enum RouteBackend<'a> {
    LocalLaya(&'a LayaSelector),
    TypeSafeJev(&'a Path),
    /// Score the 0G catalog on the host. No selector model is called.
    HostPolicy,
}

/// The composer marks an unpinned, new task with `jevAuto`. This host-only
/// option is removed before the request reaches an ACP coding provider.
pub fn requested(request: &RunRequest) -> bool {
    request
        .model_options
        .get("jevAuto")
        .and_then(|v| v.as_bool())
        == Some(true)
        && request
            .model_options
            .get("jevRouted")
            .and_then(|v| v.as_bool())
            != Some(true)
        // Model options such as Fast and Plan belong to the originally
        // selected provider. Never discard or transplant them to another.
        && request
            .model_options
            .keys()
            .all(|key| key == "jevAuto" || key == "jevRouted" || key.starts_with("og"))
}

/// Route one new coding task. Every error or abstention leaves the normal
/// Avid-selected harness untouched. The caller must recheck the returned
/// route before execution and must not reroute a live or resumed session.
pub async fn choose(
    registry: &HarnessRegistry,
    request: &RunRequest,
    backend: RouteBackend<'_>,
) -> Option<RouteChoice> {
    choose_with_activity(registry, request, backend, |_| {}).await
}

/// The callback marks only the selector's evaluation window, after the host
/// has prepared a nonempty candidate set and before its result is validated.
pub async fn choose_with_activity(
    registry: &HarnessRegistry,
    request: &RunRequest,
    backend: RouteBackend<'_>,
    activity: impl FnMut(bool),
) -> Option<RouteChoice> {
    choose_with_report(registry, request, backend, activity)
        .await
        .choice
}

/// Return the receipt alongside the route so the caller can write the user
/// message first, then place the decision in the transcript in causal order.
pub async fn choose_with_report(
    _registry: &HarnessRegistry,
    request: &RunRequest,
    backend: RouteBackend<'_>,
    mut activity: impl FnMut(bool),
) -> RouteDecision {
    if !requested(request) {
        return RouteDecision::skipped();
    }
    let prepared = prepare_routes(request).await;
    if prepared.is_empty() {
        tracing::info!("Jev router: no eligible installed models; using selected harness");
        return RouteDecision::skipped();
    }
    let state = TaskState::new(crate::now_ms().max(0) as u64);
    let Some(fingerprint) = route_fingerprint(&prepared) else {
        return RouteDecision::skipped();
    };
    let actions = prepared
        .iter()
        .map(|route| PreparedAction {
            id: route.id.clone(),
            description: route.description.clone(),
            payload: route.choice.clone(),
            prepared_at_revision: state.revision,
            preconditions: Vec::new(),
            read_set_fingerprint: Some(fingerprint.clone()),
            expires_at_ms: Some(crate::now_ms().saturating_add(45_000)),
        })
        .collect::<Vec<_>>();
    let initial_context = ValidationContext {
        now_ms: crate::now_ms(),
        current_read_set_fingerprint: Some(fingerprint.clone()),
        authorized: true,
    };
    if !actions
        .iter()
        .all(|action| workflow::eligible(&state, action, &initial_context))
    {
        return RouteDecision::skipped();
    }
    let (
        backend_name,
        considered,
        selected,
        selector_fallback,
        confidence,
        selected_probability,
        fit,
    ) = match backend {
        RouteBackend::TypeSafeJev(key_path) => {
            let Ok(selector) = JevSelector::new(JevConfig::app_owned(key_path.to_path_buf()))
            else {
                return RouteDecision::skipped();
            };
            let considered = prepared
                .iter()
                .map(|route| DecisionCandidate::new(&route.id, &route.description))
                .collect();
            let input = SelectionInput {
                state: DecisionState {
                    task: request.prompt.chars().take(6_000).collect(),
                    context: format!(
                        "Coding task. Sandbox: {:?}. Image attachments: {}. Select one 0G model id. The host sets provider strategy and 0G Router chooses the provider.",
                        request.sandbox,
                        request.attachments.len()
                    ),
                    state_version: state.revision,
                },
                candidates: prepared
                    .iter()
                    .map(|route| Candidate {
                        id: route.id.clone(),
                        description: route.description.clone(),
                    })
                    .collect(),
            };
            activity(true);
            let outcome = selector.select(input, &DecisionBudget::new(1)).await;
            activity(false);
            tracing::info!(trace = ?outcome.trace, "TypeSafe Jev route decision");
            (
                DecisionBackend::Jev,
                considered,
                outcome.selected_id,
                outcome.trace.fallback.map(|reason| format!("{reason:?}")),
                outcome.trace.confidence,
                None,
                outcome.trace.fit,
            )
        }
        RouteBackend::LocalLaya(selector) => {
            // Laya's question-prefix budget is smaller than its 1024-token
            // context. Interleave providers so eight compact options give
            // every installed agent a chance before adding second models.
            let local = compact_local_routes(&prepared);
            let considered = local
                .iter()
                .map(|route| {
                    DecisionCandidate::new(
                        &route.id,
                        format!("{}: {}", route.provider_name, route.model_label),
                    )
                })
                .collect();
            let input = SelectionInput {
                state: DecisionState {
                    task: request.prompt.chars().take(900).collect(),
                    context: format!(
                        "Choose a 0G model. Sandbox {:?}; {} attachments. Host enforces permissions.",
                        request.sandbox,
                        request.attachments.len()
                    ),
                    state_version: state.revision,
                },
                candidates: local
                    .iter()
                    .map(|route| Candidate {
                        id: route.id.clone(),
                        description: format!("{}: {}", route.provider_name, route.model_label)
                            .chars()
                            .take(90)
                            .collect(),
                    })
                    .collect(),
            };
            activity(true);
            let outcome = selector.select(input).await;
            activity(false);
            tracing::info!(trace = ?outcome.trace, "Local Laya route decision");
            (
                DecisionBackend::Laya,
                considered,
                outcome.selected_id,
                outcome.trace.fallback.map(|reason| format!("{reason:?}")),
                outcome.trace.confidence,
                outcome.trace.selected_probability,
                None,
            )
        }
        RouteBackend::HostPolicy => {
            let considered = prepared
                .iter()
                .map(|route| DecisionCandidate::new(&route.id, &route.description))
                .collect();
            let selected = prepared.first().map(|route| route.id.clone());
            let confidence = prepared.first().map(|route| route.confidence);
            (
                DecisionBackend::Host,
                considered,
                selected,
                None,
                confidence,
                None,
                None,
            )
        }
    };
    let mut event = DecisionEvent::new(
        format!("intake-{}", state.revision),
        state.revision,
        backend_name,
        DecisionStage::Intake,
        considered,
        match &selected {
            Some(candidate_id) => DecisionResult::Selected {
                candidate_id: candidate_id.clone(),
            },
            None => DecisionResult::Abstained,
        },
        DecisionValidation::Accepted,
        crate::now_ms(),
    );
    event.confidence = confidence;
    event.selected_probability = selected_probability;
    event.fit = fit;
    event = event.bounded();
    if let Some(reason) = selector_fallback {
        event = event.with_fallback(reason);
    }
    let Some(selected) = selected else {
        return scored_fallback(
            &prepared,
            event,
            "Selector abstained; host scored the 0G catalog",
        );
    };
    let Some(action) = actions.iter().find(|action| action.id == selected) else {
        event.validation = DecisionValidation::Rejected;
        return scored_fallback(&prepared, event, "Unknown route ID");
    };
    let live = prepare_routes(request).await;
    let Some(live_fingerprint) = route_fingerprint(&live) else {
        event.validation = DecisionValidation::Stale;
        return scored_fallback(&live, event, "Route catalog unavailable");
    };
    let still_present = live.iter().any(|route| route.id == action.id);
    let validation = workflow::validate_selected(
        &state,
        action,
        &selected,
        &ValidationContext {
            now_ms: crate::now_ms(),
            current_read_set_fingerprint: Some(live_fingerprint),
            authorized: still_present,
        },
    );
    if !validation.accepted {
        tracing::info!(trace = ?validation, "Decision route became ineligible");
        event.validation = match validation.rejection {
            Some(workflow::RejectReason::Expired) => DecisionValidation::Expired,
            Some(workflow::RejectReason::Unauthorized) => DecisionValidation::Unauthorized,
            Some(workflow::RejectReason::StaleReadSet | workflow::RejectReason::StaleRevision) => {
                DecisionValidation::Stale
            }
            _ => DecisionValidation::Rejected,
        };
        return scored_fallback(&live, event, "Route changed before dispatch");
    }
    scheduled(action.payload.clone(), event)
}

#[cfg(test)]
fn reasoning_compatible(requested: Option<ReasoningLevel>, offered: &[ReasoningLevel]) -> bool {
    requested.is_none_or(|level| offered.contains(&level))
}

fn route_fingerprint(routes: &[PreparedRoute]) -> Option<String> {
    workflow::fingerprint_of(
        &routes
            .iter()
            .map(|route| {
                (
                    &route.id,
                    &route.description,
                    &route.choice.harness,
                    &route.choice.model,
                    &route.reasoning_levels,
                )
            })
            .collect::<Vec<_>>(),
    )
    .ok()
}

fn compact_local_routes(prepared: &[PreparedRoute]) -> Vec<&PreparedRoute> {
    prepared.iter().take(MAX_LOCAL_ROUTES).collect()
}

fn objective_of(request: &RunRequest) -> dsh_harness_bridge::og::Objective {
    match request
        .model_options
        .get("ogObjective")
        .and_then(|value| value.as_str())
    {
        Some("cost") => dsh_harness_bridge::og::Objective::Cost,
        Some("speed") => dsh_harness_bridge::og::Objective::Speed,
        Some("quality") => dsh_harness_bridge::og::Objective::Quality,
        _ => dsh_harness_bridge::og::objective_from_env(),
    }
}

fn trust_of(request: &RunRequest) -> dsh_harness_bridge::og::TrustMode {
    match request
        .model_options
        .get("ogTrust")
        .and_then(|value| value.as_str())
    {
        Some("private") => dsh_harness_bridge::og::TrustMode::Private,
        Some("verified") => dsh_harness_bridge::og::TrustMode::Verified,
        Some("standard") => dsh_harness_bridge::og::TrustMode::Standard,
        _ => dsh_harness_bridge::og::trust_from_env(),
    }
}

async fn prepare_routes(request: &RunRequest) -> Vec<PreparedRoute> {
    let objective = objective_of(request);
    let trust = trust_of(request);
    let strategy = dsh_harness_bridge::og::provider_strategy(objective);
    let verify = dsh_harness_bridge::og::verify_tee(trust);
    let catalog = dsh_harness_bridge::og::load_catalog().await;
    let ranked = dsh_harness_bridge::og::rank(&request.prompt, &catalog, objective, trust);
    ranked
        .into_iter()
        .take(MAX_ROUTES)
        .map(|model| {
            let description = format!(
                "{}; capabilities={}; context={}; providers={}; trust={}; promptPrice={}",
                model.name,
                if model.capabilities.is_empty() {
                    "general".to_string()
                } else {
                    model.capabilities.join(",")
                },
                model.context_length,
                model.provider_count,
                model.verifiability,
                model.prompt_price,
            );
            let label = model.name.clone();
            let id = model.id.clone();
            PreparedRoute {
                id: id.clone(),
                choice: RouteChoice {
                    harness: HarnessId::Og,
                    model: Some(id),
                    provider_strategy: strategy.to_string(),
                    trust_mode: match trust {
                        dsh_harness_bridge::og::TrustMode::Private => "private",
                        dsh_harness_bridge::og::TrustMode::Verified => "verified",
                        dsh_harness_bridge::og::TrustMode::Standard => "standard",
                    }
                    .to_string(),
                    verify_tee: verify,
                },
                description,
                provider_name: "0G".into(),
                model_label: label,
                reasoning_levels: vec![ReasoningLevel::Medium],
                confidence: 0.75,
            }
        })
        .collect()
}

fn scheduled(choice: RouteChoice, event: DecisionEvent) -> RouteDecision {
    let outcome = format!(
        "0G scheduled model={} strategy={} trust={}",
        choice.model.as_deref().unwrap_or("unknown"),
        choice.provider_strategy,
        choice.trust_mode
    );
    RouteDecision {
        choice: Some(choice),
        event: Some(event.with_observed_outcome(outcome)),
    }
}

fn scored_fallback(
    prepared: &[PreparedRoute],
    event: DecisionEvent,
    reason: &str,
) -> RouteDecision {
    let Some(route) = prepared.first() else {
        return RouteDecision {
            choice: None,
            event: Some(
                event
                    .with_fallback(reason)
                    .with_observed_outcome("0G catalog empty"),
            ),
        };
    };
    let mut event = event.with_fallback(reason);
    event.result = DecisionResult::Selected {
        candidate_id: route.id.clone(),
    };
    event.validation = DecisionValidation::Accepted;
    scheduled(route.choice.clone(), event)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_snapshot_changes_when_a_model_changes() {
        let mut routes = vec![PreparedRoute {
            id: "route_0".into(),
            choice: RouteChoice {
                harness: HarnessId::Codex,
                model: Some("model-a".into()),
                provider_strategy: "default".into(),
                trust_mode: "standard".into(),
                verify_tee: false,
            },
            description: "Codex model A".into(),
            provider_name: "Codex".into(),
            model_label: "A".into(),
            reasoning_levels: vec![ReasoningLevel::High],
            confidence: 0.75,
        }];
        let original = route_fingerprint(&routes).unwrap();
        routes[0].choice.model = Some("model-b".into());
        assert_ne!(route_fingerprint(&routes).unwrap(), original);
    }

    #[test]
    fn requested_reasoning_must_exist_on_routed_model() {
        assert!(reasoning_compatible(None, &[]));
        assert!(reasoning_compatible(
            Some(ReasoningLevel::High),
            &[ReasoningLevel::High],
        ));
        assert!(!reasoning_compatible(
            Some(ReasoningLevel::High),
            &[ReasoningLevel::Medium],
        ));
    }

    #[test]
    fn only_new_auto_tasks_are_routed() {
        let mut request = RunRequest {
            prompt: "fix test".into(),
            harness: Some(HarnessId::Codex),
            model: None,
            reasoning: None,
            model_options: Default::default(),
            cwd: "/tmp".into(),
            sandbox: keel_proto::SandboxLevel::WorkspaceWrite,
            auto_approve: false,
            resume: None,
            attachments: Vec::new(),
        };
        assert!(!requested(&request));
        request.model_options.insert("jevAuto".into(), true.into());
        assert!(requested(&request));
        request
            .model_options
            .insert("ogObjective".into(), "cost".into());
        assert!(requested(&request));
        request.model_options.remove("ogObjective");
        request.model_options.insert("fast".into(), true.into());
        assert!(!requested(&request));
        request.model_options.remove("fast");
        request
            .model_options
            .insert("jevRouted".into(), true.into());
        assert!(!requested(&request));
    }
}

//! The dsh harness reaches the app's catalog: `ListHarnesses` (what the
//! composer picker and Settings → Agents render) must show the in-process
//! DeepSeek Harness, describe it correctly, and resolve it without a CLI.

use keel_engine::default_registry;
use keel_engine::registry::descriptor_enabled;
use keel_proto::{HarnessId, SteeringMode};

#[test]
fn dsh_harness_is_catalogued_and_resolvable() {
    let registry = default_registry();
    let descriptors = registry.descriptors();
    let dsh = descriptors
        .iter()
        .find(|descriptor| descriptor.id == HarnessId::Dsh)
        .expect("DeepSeek Harness appears in the catalog");

    assert_eq!(dsh.name, "DeepSeek Harness");
    assert!(dsh.supports_steering);
    assert_eq!(dsh.steering_mode, SteeringMode::StepBoundary);
    // In-process, but only ready in the picker with a DeepSeek credential.
    // An OG_API_KEY must not mark this slot installed.
    assert_eq!(
        dsh.installed,
        dsh_harness_bridge::deepseek_credential_available()
    );
    // Opt-in like every non-default harness (Claude Code / Codex ship on).
    assert!(!descriptor_enabled(dsh));

    // The lazy slot resolves to a live harness without spawning anything.
    let harness = registry
        .resolve(HarnessId::Dsh)
        .expect("dsh slot resolves in-process");
    assert_eq!(harness.id(), HarnessId::Dsh);
    assert_eq!(harness.display_name(), "DeepSeek Harness");

    let og = descriptors
        .iter()
        .find(|descriptor| descriptor.id == HarnessId::Og)
        .expect("0G Router appears beside DeepSeek");
    assert_eq!(og.name, "0G Router");
    assert_eq!(og.installed, dsh_harness_bridge::og::api_key_available());
    assert!(!descriptor_enabled(og));
    let og_harness = registry
        .resolve(HarnessId::Og)
        .expect("0G slot resolves in-process");
    assert_eq!(og_harness.id(), HarnessId::Og);
    assert_eq!(og_harness.display_name(), "0G Router");

    // The descriptor must not drift from the resolved harness (the same
    // stability rule the claude/codex slots are held to).
    assert_eq!(dsh.supports_steering, harness.supports_steering());
    assert_eq!(dsh.steering_mode, harness.steering_mode());
    assert_eq!(dsh.reasoning_levels, harness.reasoning_levels().to_vec());
}

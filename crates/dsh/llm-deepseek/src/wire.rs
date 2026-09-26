//! DeepSeek chat-completions wire format (OpenAI-compatible), ported from
//! `packages/llm/llm-deepseek/src/types.ts`. Types only.

use serde::{Deserialize, Serialize};

/// Request body for `POST {base_url}/chat/completions`.
#[derive(Debug, Clone, Serialize)]
pub struct WireRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    /// Always streaming.
    pub stream: bool,
    pub stream_options: WireStreamOptions,
    /// Thinking-mode toggle (top level on the wire).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<WireThinking>,
    /// Thinking effort; low/medium map to high server-side, so only
    /// high/max ride the wire.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Stop sequences (OpenAI `stop`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Ask 0G Router to verify a TEE attestation for this completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verify_tee: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireStreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireThinking {
    /// `"enabled"` or `"disabled"`.
    #[serde(rename = "type")]
    pub kind: String,
}

/// One entry of the request `messages` array, discriminated on `role`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum WireMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    /// Assistant history. Text-less tool-call turns send `""`, never null —
    /// some gateways reject null, and the live API rejects assistant
    /// messages with neither content nor tool calls. `reasoning_content` is
    /// the thinking-mode CoT passback, required on tool-call turns and
    /// dropped on plain turns to save tokens.
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<WireToolCall>>,
    },
    /// The result of one tool call, keyed by its call id.
    Tool {
        tool_call_id: String,
        content: String,
    },
}

/// A completed tool call replayed on an assistant history message;
/// `arguments` is the raw JSON string.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WireToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: WireFunctionCall,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WireFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// One entry of the request `tools` array; `parameters` is a JSON Schema
/// object.
#[derive(Debug, Clone, Serialize)]
pub struct WireTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: WireFunctionDef,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireFunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

/// One parsed SSE `data:` payload (a chat.completion.chunk).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireChunk {
    #[serde(default)]
    pub choices: Vec<WireChoice>,
    /// Arrives attached to the finish chunk and/or as a trailing usage-only
    /// chunk.
    #[serde(default)]
    pub usage: Option<WireUsage>,
}

/// One streamed choice; `finish_reason` is non-null only on its terminal
/// chunk.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireChoice {
    #[serde(default)]
    pub delta: Option<WireDelta>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// The incremental content of one streamed choice; any subset per chunk.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireDelta {
    /// Visible text; null/empty on reasoning and tool-call chunks.
    #[serde(default)]
    pub content: Option<String>,
    /// Thinking-mode CoT; the FIRST chunk carries an empty string (must not
    /// open a reasoning block) and is absent entirely in non-thinking mode.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<WireToolCallDelta>,
}

/// A streamed fragment of one tool call; fragments sharing an `index`
/// concatenate into one call.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireToolCallDelta {
    /// Disambiguates parallel tool calls; stable across a call's deltas.
    pub index: u64,
    /// Present on the first delta of each call only.
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<WireFunctionDelta>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireFunctionDelta {
    /// Present on the first delta of each call only.
    #[serde(default)]
    pub name: Option<String>,
    /// Argument JSON fragment (concatenate across deltas).
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Wire token accounting. `prompt_tokens` INCLUDES cache hits; the harness
/// convention is disjoint counts, so translation subtracts them out.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u64>,
    #[serde(default)]
    pub prompt_tokens_details: Option<WirePromptDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<WireCompletionDetails>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WirePromptDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireCompletionDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

/// Non-2xx error body.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireError {
    #[serde(default)]
    pub error: Option<WireErrorDetail>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireErrorDetail {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default, rename = "type")]
    pub error_type: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
}

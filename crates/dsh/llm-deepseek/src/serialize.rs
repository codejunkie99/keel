//! Serialize harness messages into DeepSeek chat completions, ported from
//! `packages/llm/llm-deepseek/src/serialize.ts`. User text joins; assistant
//! text becomes `content`, tool calls become `tool_calls`, tool results
//! become standalone tool-role messages. Assistant reasoning replays as
//! `reasoning_content` only on tool-call turns (thinking-mode passback).
//! Image blocks are rejected: this wire route is text-only.

use crate::wire::{
    WireFunctionCall, WireFunctionDef, WireMessage, WireRequest, WireStreamOptions, WireThinking,
    WireTool, WireToolCall,
};
use dsh_llm::{
    CallPurpose, ContentBlock, GenerateOptions, LlmError, Message, Role, content_has_image,
};

/// Adapter-level request defaults (from plugin config).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestDefaults {
    /// `Some("enabled")` / `Some("disabled")`; `None` puts nothing on the wire.
    pub thinking: Option<String>,
    /// `off` / `high` / `max`.
    pub reasoning_effort: Option<String>,
}

struct ResolvedThinking {
    thinking: Option<String>,
    reasoning_effort: Option<String>,
}

fn validated_effort(effort: &str) -> Result<String, LlmError> {
    match effort {
        "off" | "high" | "max" => Ok(effort.to_string()),
        other => Err(LlmError::new(
            format!("DeepSeek does not support reasoning effort \"{other}\""),
            "UNSUPPORTED_REASONING_EFFORT",
        )),
    }
}

/// Resolve one legal thinking/effort pair without exposing `off` as a wire
/// effort. A session-title call always disables thinking (a short title
/// budget must produce visible text).
fn resolve_thinking(
    options: &GenerateOptions,
    defaults: &RequestDefaults,
) -> Result<ResolvedThinking, LlmError> {
    if options.purpose == Some(CallPurpose::SessionTitle) {
        return Ok(ResolvedThinking {
            thinking: Some("disabled".into()),
            reasoning_effort: None,
        });
    }
    let effort = match &options.reasoning_effort {
        Some(effort) => Some(validated_effort(effort.as_str())?),
        None => defaults.reasoning_effort.clone(),
    };
    if defaults.thinking.as_deref() == Some("disabled") {
        if let Some(effort) = &effort {
            if effort != "off" {
                return Err(LlmError::new(
                    format!("DeepSeek deployment does not support reasoning effort \"{effort}\""),
                    "UNSUPPORTED_REASONING_EFFORT",
                ));
            }
        }
    }
    Ok(match effort.as_deref() {
        Some("off") => ResolvedThinking {
            thinking: Some("disabled".into()),
            reasoning_effort: None,
        },
        Some(effort @ ("high" | "max")) => ResolvedThinking {
            thinking: Some("enabled".into()),
            reasoning_effort: Some(effort.to_string()),
        },
        _ => ResolvedThinking {
            thinking: defaults.thinking.clone(),
            reasoning_effort: None,
        },
    })
}

/// Join the text blocks of a message (user/tool-result content).
fn flatten_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Reject image content before any text-flattening path can silently erase it.
fn assert_text_only(blocks: &[ContentBlock]) -> Result<(), LlmError> {
    if content_has_image(blocks) {
        return Err(LlmError::new(
            "The DeepSeek chat-completions adapter does not support image content.",
            "UNSUPPORTED_CONTENT",
        ));
    }
    Ok(())
}

fn serialize_assistant(message: &Message) -> WireMessage {
    let text = flatten_text(&message.content);
    let reasoning: String = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Reasoning { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let tool_calls: Vec<WireToolCall> = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => Some(WireToolCall {
                id: id.as_str().to_string(),
                kind: "function".into(),
                function: WireFunctionCall {
                    name: name.clone(),
                    arguments: arguments.clone(),
                },
            }),
            _ => None,
        })
        .collect();
    WireMessage::Assistant {
        content: text,
        reasoning_content: (!tool_calls.is_empty() && !reasoning.is_empty()).then_some(reasoning),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
    }
}

/// Serialize the conversation, order preserved: harness tool results ride in
/// user messages but become standalone tool-role wire messages (a mixed user
/// message contributes its text first, its results after).
pub fn serialize_messages(messages: &[Message]) -> Result<Vec<WireMessage>, LlmError> {
    let mut wire = Vec::new();
    for message in messages {
        assert_text_only(&message.content)?;
        match message.role {
            Role::System => {
                wire.push(WireMessage::System {
                    content: flatten_text(&message.content),
                });
            }
            Role::Assistant => {
                wire.push(serialize_assistant(message));
            }
            Role::User => {
                let tool_results: Vec<(&dsh_llm::CallId, &Vec<ContentBlock>)> = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolResult {
                            tool_call_id,
                            content,
                            ..
                        } => Some((tool_call_id, content)),
                        _ => None,
                    })
                    .collect();
                let text = flatten_text(&message.content);
                if !text.is_empty() || tool_results.is_empty() {
                    wire.push(WireMessage::User { content: text });
                }
                for (call_id, content) in tool_results {
                    let flattened = flatten_text(content);
                    wire.push(WireMessage::Tool {
                        tool_call_id: call_id.as_str().to_string(),
                        // Empty tool output still needs SOME wire content.
                        content: if flattened.is_empty() {
                            "(no output)".to_string()
                        } else {
                            flattened
                        },
                    });
                }
            }
        }
    }
    Ok(wire)
}

/// Build the full wire request: always streaming with usage reporting on;
/// optional fields are omitted rather than sent as null so provider defaults
/// apply.
pub fn serialize_request(
    options: &GenerateOptions,
    defaults: &RequestDefaults,
) -> Result<WireRequest, LlmError> {
    let mut messages = Vec::new();
    if let Some(system) = &options.system {
        messages.push(WireMessage::System {
            content: system.clone(),
        });
    }
    messages.extend(serialize_messages(&options.messages)?);

    let tools: Option<Vec<WireTool>> = options.tools.as_ref().and_then(|tools| {
        if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|tool| WireTool {
                        kind: "function".into(),
                        function: WireFunctionDef {
                            name: tool.name.clone(),
                            description: tool.description.clone(),
                            parameters: tool.parameters.clone(),
                        },
                    })
                    .collect(),
            )
        }
    });
    let resolved = resolve_thinking(options, defaults)?;

    Ok(WireRequest {
        model: options.model.clone(),
        messages,
        stream: true,
        stream_options: WireStreamOptions {
            include_usage: true,
        },
        thinking: resolved.thinking.map(|kind| WireThinking { kind }),
        reasoning_effort: resolved.reasoning_effort,
        tools,
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        stop: options.stop.clone(),
        verify_tee: None,
    })
}

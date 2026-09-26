//! Translate DeepSeek SSE payloads into harness stream chunks, ported from
//! `packages/llm/llm-deepseek/src/translate.ts`: one stateful block per
//! content, reasoning, or tool-call index; an empty initial reasoning delta
//! does not open a block; block-ends, usage, and finish all defer to the
//! `[DONE]` sentinel so no chunk ever follows `finish`.

use crate::wire::{WireChunk, WireUsage};
use dsh_llm::{
    BlockType, CallId, ContentBlock, EMPTY_RESPONSE_CODE, FinishReason, LlmError, LlmFailure,
    StreamChunk, TokenUsage,
};

#[derive(Clone, Copy, PartialEq)]
enum OpenKind {
    Text,
    Reasoning,
    ToolCall,
}

struct OpenBlock {
    index: u64,
    kind: OpenKind,
    text: String,
    call_id: Option<String>,
    name: Option<String>,
}

/// Map the wire `finish_reason` vocabulary; unrecognized values
/// (content_filter, …) become an error finish with the uppercased value as
/// its code.
pub fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "tool_calls" => FinishReason::ToolCalls,
        "length" => FinishReason::MaxTokens,
        other => FinishReason::Error {
            failure: LlmFailure::new(format!("model stopped: {other}"), other.to_uppercase()),
        },
    }
}

/// Map wire usage. DeepSeek's `prompt_tokens` INCLUDES cache hits; the
/// harness convention is disjoint counts, so cache reads subtract out of
/// `input_tokens`.
pub fn map_usage(usage: &WireUsage) -> TokenUsage {
    let cache_read = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .or(usage.prompt_cache_hit_tokens);
    let reasoning = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens);
    TokenUsage {
        input_tokens: usage.prompt_tokens.saturating_sub(cache_read.unwrap_or(0)),
        output_tokens: usage.completion_tokens,
        cache_read_tokens: cache_read,
        cache_write_tokens: None,
        reasoning_tokens: reasoning,
    }
}

fn close_block(block: &OpenBlock) -> ContentBlock {
    match block.kind {
        OpenKind::Text => ContentBlock::Text {
            text: block.text.clone(),
        },
        OpenKind::Reasoning => ContentBlock::Reasoning {
            text: block.text.clone(),
        },
        OpenKind::ToolCall => ContentBlock::ToolCall {
            id: CallId::new(block.call_id.clone().unwrap_or_default()),
            name: block.name.clone().unwrap_or_default(),
            arguments: block.text.clone(),
        },
    }
}

/// Stateful payload-to-chunk translator (the upstream generator's state made
/// explicit). Feed each SSE data payload; `done` handles the `[DONE]`
/// sentinel.
#[derive(Default)]
pub struct Translator {
    next_index: u64,
    text_slot: Option<usize>,
    reasoning_slot: Option<usize>,
    tool_slots: std::collections::HashMap<u64, usize>,
    order: Vec<OpenBlock>,
    pending_finish: Option<FinishReason>,
    pending_usage: Option<TokenUsage>,
}

impl Translator {
    pub fn new() -> Self {
        Self::default()
    }

    fn open(&mut self, kind: OpenKind) -> usize {
        let block = OpenBlock {
            index: self.next_index,
            kind,
            text: String::new(),
            call_id: None,
            name: None,
        };
        self.next_index += 1;
        self.order.push(block);
        self.order.len() - 1
    }

    /// Translate one (non-`[DONE]`) SSE data payload into stream chunks.
    /// Malformed JSON aborts the stream with `MALFORMED_RESPONSE`.
    pub fn feed(&mut self, payload: &str) -> Result<Vec<StreamChunk>, LlmError> {
        let chunk: WireChunk = serde_json::from_str(payload).map_err(|_| {
            let head: String = payload.chars().take(120).collect();
            LlmError::new(
                format!("malformed SSE payload: {head}"),
                "MALFORMED_RESPONSE",
            )
        })?;
        let mut out = Vec::new();
        for choice in &chunk.choices {
            let Some(delta) = &choice.delta else {
                if let Some(reason) = &choice.finish_reason {
                    self.pending_finish = Some(map_finish_reason(reason));
                }
                continue;
            };

            // Reasoning first: thinking mode interleaves it before text; the
            // empty-string first chunk must not open a block.
            if let Some(reasoning) = &delta.reasoning_content {
                if !reasoning.is_empty() {
                    let slot = match self.reasoning_slot {
                        Some(slot) => slot,
                        None => {
                            let slot = self.open(OpenKind::Reasoning);
                            self.reasoning_slot = Some(slot);
                            out.push(StreamChunk::BlockStart {
                                index: self.order[slot].index,
                                block_type: BlockType::Reasoning,
                            });
                            slot
                        }
                    };
                    self.order[slot].text.push_str(reasoning);
                    out.push(StreamChunk::ReasoningDelta {
                        index: self.order[slot].index,
                        text: reasoning.clone(),
                    });
                }
            }

            if let Some(content) = &delta.content {
                if !content.is_empty() {
                    let slot = match self.text_slot {
                        Some(slot) => slot,
                        None => {
                            let slot = self.open(OpenKind::Text);
                            self.text_slot = Some(slot);
                            out.push(StreamChunk::BlockStart {
                                index: self.order[slot].index,
                                block_type: BlockType::Text,
                            });
                            slot
                        }
                    };
                    self.order[slot].text.push_str(content);
                    out.push(StreamChunk::TextDelta {
                        index: self.order[slot].index,
                        text: content.clone(),
                    });
                }
            }

            for call in &delta.tool_calls {
                let slot = match self.tool_slots.get(&call.index) {
                    Some(slot) => *slot,
                    None => {
                        let slot = self.open(OpenKind::ToolCall);
                        self.tool_slots.insert(call.index, slot);
                        out.push(StreamChunk::BlockStart {
                            index: self.order[slot].index,
                            block_type: BlockType::ToolCall,
                        });
                        slot
                    }
                };
                if let Some(id) = &call.id {
                    self.order[slot].call_id = Some(id.clone());
                }
                if let Some(name) = call.function.as_ref().and_then(|f| f.name.clone()) {
                    self.order[slot].name = Some(name);
                }
                let fragment = call
                    .function
                    .as_ref()
                    .and_then(|f| f.arguments.clone())
                    .unwrap_or_default();
                self.order[slot].text.push_str(&fragment);
                out.push(StreamChunk::ToolCallDelta {
                    index: self.order[slot].index,
                    id: CallId::new(self.order[slot].call_id.clone().unwrap_or_default()),
                    name: self.order[slot].name.clone(),
                    arguments_delta: fragment,
                });
            }

            if let Some(reason) = &choice.finish_reason {
                self.pending_finish = Some(map_finish_reason(reason));
            }
        }
        // Usage may ride the finish chunk or a trailing usage-only chunk —
        // keep the latest.
        if let Some(usage) = &chunk.usage {
            self.pending_usage = Some(map_usage(usage));
        }
        Ok(out)
    }

    /// True once a choice payload opened a block or named a finish reason.
    /// A later clean EOF can then be flushed like `[DONE]`.
    pub fn has_terminal_progress(&self) -> bool {
        self.pending_finish.is_some() || !self.order.is_empty()
    }

    /// Handle the `[DONE]` sentinel: emit deferred block-ends, usage, and the
    /// finish. A `stop` (or absent) finish with no opened blocks is a
    /// degenerate provider completion → `EMPTY_RESPONSE` error finish.
    pub fn done(&mut self) -> Vec<StreamChunk> {
        let mut out = Vec::new();
        for block in &self.order {
            out.push(StreamChunk::BlockEnd {
                index: block.index,
                block: close_block(block),
            });
        }
        if let Some(usage) = self.pending_usage.take() {
            out.push(StreamChunk::Usage { usage });
        }
        let reason = self.pending_finish.take().unwrap_or(FinishReason::Stop);
        let reason = if matches!(reason, FinishReason::Stop) && self.order.is_empty() {
            FinishReason::Error {
                failure: LlmFailure::new(
                    "model returned a completed response with no content",
                    EMPTY_RESPONSE_CODE,
                ),
            }
        } else {
            reason
        };
        out.push(StreamChunk::Finish {
            reason,
            replay_state: None,
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::Translator;

    #[test]
    fn content_counts_as_terminal_progress() {
        let mut translator = Translator::new();
        assert!(!translator.has_terminal_progress());
        translator
            .feed(r#"{"choices":[{"delta":{"content":"hi"}}]}"#)
            .unwrap();
        assert!(translator.has_terminal_progress());
    }
}

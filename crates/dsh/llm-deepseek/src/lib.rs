//! Rust port of `@deepseek-ai/dsh-llm-deepseek` (`packages/llm/llm-deepseek`):
//! the direct-fetch DeepSeek (OpenAI-compatible) chat-completions adapter —
//! wire types, request serialization with thinking-mode rules, SSE framing,
//! chunk translation, and the `LlmAdapter` implementation.
//!
//! Crate-level divergences (modules document their own):
//! - The registering plugin (`index.ts`: settings/credentials wiring and the
//!   configurable-provider directory) is ported separately with the bundle
//!   tier; this crate is the adapter itself plus its constructor hooks.
//! - `eventsource-parser` is replaced by an in-crate SSE frame decoder.

mod adapter;
mod serialize;
mod sse;
mod translate;
mod wire;

pub use adapter::{
    DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS, DEFAULT_STREAM_IDLE_TIMEOUT_MS, DeepSeekAdapter,
    DeepSeekAdapterOptions, DeepSeekCatalogModel, DeepSeekConnectionOptions, RequestOverlay,
    http_error_code,
};
pub use serialize::{RequestDefaults, serialize_messages, serialize_request};
pub use sse::{DONE, SseDecoder};
pub use translate::{Translator, map_finish_reason, map_usage};
pub use wire::*;

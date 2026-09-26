//! The `dsh` CLI: the Rust composition of upstream `apps/cli` +
//! `packages/bundle/headless` — mount the core services, register the
//! DeepSeek adapter from the environment, run one task through the agent
//! loop, and stream the answer to stdout.
//!
//! Divergences:
//! - Composition is compile-time Rust (no cordis.yml loader / patch layers /
//!   `dsh plugin` forwarder yet); `--profile` is reserved.

use clap::Parser;
use dsh_agent::{AgentOptions, AgentRegistry, CreateAgentOptions, ResumeAgentOptions};
use dsh_agent_loop::AgentLoop;
use dsh_cordis::App;
use dsh_llm::{ContentBlock, LlmRuntime, MessageSource, StreamChunk, assert_usable_api_key};
use dsh_llm_deepseek::{
    DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS, DEFAULT_STREAM_IDLE_TIMEOUT_MS, DeepSeekAdapter,
    DeepSeekAdapterOptions, DeepSeekConnectionOptions, RequestDefaults,
};
use dsh_session::{SessionEventData, SessionId, SessionMeta, SessionStore};
use dsh_session_persistence::{PersistenceService, SessionPersistence, install_write_behind};
use dsh_session_persistence_jsonl::JsonlPersistence;
use dsh_system_prompt::{Config as PromptConfig, SystemPrompt};
use dsh_tools::{Config as ToolsConfig, ToolRuntime};
use futures::FutureExt;
use std::io::Write;
use std::rc::Rc;

/// DeepSeek Harness (Rust port): run one task through the agent loop.
#[derive(Parser)]
#[command(name = "dsh", version, about)]
struct Cli {
    /// The task to run; omit with --list to browse sessions.
    task: Option<String>,
    /// Provider route (must have a registered adapter).
    #[arg(long, default_value = "deepseek")]
    provider: String,
    /// Model id sent to the provider.
    #[arg(long, default_value = "deepseek-chat")]
    model: String,
    /// Resume a persisted session by id instead of creating one.
    #[arg(long)]
    resume: Option<String>,
    /// List persisted sessions and exit.
    #[arg(long)]
    list: bool,
    /// Base URL of the chat-completions endpoint (overrides
    /// $DEEPSEEK_BASE_URL; default https://api.deepseek.com).
    #[arg(long)]
    base_url: Option<String>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    dsh_cordis::run(run(cli))
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    let app = App::new();
    let ctx = app.root();

    // Storage under $DSH_HOME (default ~/.dsh), matching upstream layout.
    let home = dsh_home_paths::resolve_dsh_home(None);
    let sessions_dir = home.join("sessions");
    let backend = Rc::new(JsonlPersistence::new(&sessions_dir)?);

    let sessions =
        SessionStore::provide(&ctx).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let agents =
        AgentRegistry::provide(&ctx).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let llm = LlmRuntime::provide(&ctx).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let system_prompt = SystemPrompt::new(
        &ctx,
        PromptConfig {
            persona: "Answer the user's task directly and concisely.".into(),
            ..PromptConfig::default()
        },
    )?;
    let tools = ToolRuntime::provide(&ctx, ToolsConfig::default())?;
    // Model-facing tools: fs read/write/edit under the read-before-edit
    // observation policy, glob/grep search, and bash. Resolution base is the
    // process cwd (a default, never a containment boundary).
    {
        let fs = dsh_fs::LocalFileSystem::provide(
            &ctx,
            dsh_fs::LocalFileSystemConfig {
                cwd: None,
                diff_basis_max_bytes: None,
            },
        )?;
        dsh_fs::install_observation_policy(&ctx)?;
        dsh_fs::register_fs_tools(&ctx, &tools, &fs, dsh_fs::FsToolsConfig::default())?;
        dsh_fs::register_search_tools(&ctx, &tools, dsh_fs::SearchToolsConfig::default())?;
        let subprocess = dsh_shell::LocalSubprocessRuntime::provide(&ctx)?;
        let shell = dsh_shell::LocalBashExecutor::provide(
            &ctx,
            subprocess,
            dsh_shell::BashConfig::default(),
        )?;
        dsh_shell::register_bash_tool(&ctx, &tools, &shell)?;
    }
    PersistenceService::provide(&ctx, backend.clone())
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    install_write_behind(&ctx, backend.clone(), 200)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    if cli.list {
        let headers = backend.list().await?;
        if headers.is_empty() {
            println!("no sessions in {}", sessions_dir.display());
        }
        for header in headers {
            println!(
                "{}\t{}",
                header.id.as_str(),
                header.cwd.as_deref().unwrap_or("-")
            );
        }
        return Ok(());
    }
    let Some(task) = cli.task.clone() else {
        anyhow::bail!("no task given (pass a task string, or --list)");
    };

    // Prompt composition: identity + persona ship with the SystemPrompt
    // config above; the tool schemas wire in here.
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

    // DeepSeek adapter: connection facts resolved per operation, credential
    // from the environment at request time (never stored, never echoed).
    let base_url = cli
        .base_url
        .clone()
        .or_else(|| std::env::var("DEEPSEEK_BASE_URL").ok())
        .unwrap_or_else(|| "https://api.deepseek.com".to_string())
        .trim_end_matches('/')
        .to_string();
    let provider_route = cli.provider.clone();
    llm.register_adapter(
        &[provider_route.clone()],
        Rc::new(DeepSeekAdapter::new(DeepSeekAdapterOptions {
            options: Box::new(move || DeepSeekConnectionOptions {
                base_url: base_url.clone(),
                api_key_env: "DEEPSEEK_API_KEY".into(),
                defaults: RequestDefaults::default(),
                max_tokens: DEFAULT_MAX_TOKENS,
                default_context_window: DEFAULT_CONTEXT_WINDOW,
                models: vec![],
                stream_idle_timeout_ms: DEFAULT_STREAM_IDLE_TIMEOUT_MS,
                retry_policy: dsh_llm::resolve_retry_policy(None, "dsh: deepseek retryPolicy")
                    .expect("default retry policy resolves"),
                extra_headers: Vec::new(),
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
        })),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    AgentLoop::install(
        &ctx,
        agents.clone(),
        sessions.clone(),
        llm,
        tools,
        system_prompt,
    )?;

    // Stream assistant text deltas to stdout as they arrive.
    ctx.on::<dsh_session::SessionEventPublished, _, _>(Default::default(), |_ctx, (_, event)| {
        if let SessionEventData::AssistantChunk { chunk, .. } = &event.data {
            match chunk {
                StreamChunk::TextDelta { text, .. } => {
                    print!("{text}");
                    let _ = std::io::stdout().flush();
                }
                StreamChunk::Finish { .. } => println!(),
                _ => {}
            }
        }
        async { None }
    })
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    let agent_options = AgentOptions {
        provider: Some(cli.provider.clone()),
        model: Some(cli.model.clone()),
        max_tokens: None,
    };
    let handle = match &cli.resume {
        Some(id) => {
            agents
                .resume(ResumeAgentOptions {
                    resume_session_id: SessionId::new(id.clone()),
                    agent_options,
                })
                .await?
        }
        None => {
            let session_id = SessionId::new(format!("session-{}", uuid::Uuid::new_v4()));
            let cwd = std::env::current_dir()
                .ok()
                .map(|path| path.to_string_lossy().into_owned());
            agents
                .create(CreateAgentOptions {
                    session_id,
                    meta: SessionMeta {
                        cwd,
                        ..Default::default()
                    },
                    seed: vec![],
                    agent_options,
                })
                .await?
        }
    };
    eprintln!("session: {}", handle.agent.id().as_str());

    handle.agent.followup(dsh_llm::create_user_message(
        vec![ContentBlock::Text { text: task }],
        MessageSource::User,
    ));
    handle.agent.when_idle().await;

    // Report a failed turn on stderr with a non-zero exit.
    let failed = handle.agent.session().with_events(|events| {
        events.iter().rev().find_map(|event| match &event.data {
            SessionEventData::TurnEnd { reason, .. } => match reason {
                dsh_session::TurnEndReason::Error { error } => {
                    Some(format!("{} ({})", error.message, error.code))
                }
                _ => None,
            },
            _ => None,
        })
    });

    // Durability checkpoint before exit.
    let session = handle.agent.session();
    sessions
        .flush(&session)
        .await
        .map_err(|error| anyhow::anyhow!(error.0))?;
    handle.dispose().await;

    if let Some(failure) = failed {
        anyhow::bail!("task failed: {failure}");
    }
    Ok(())
}

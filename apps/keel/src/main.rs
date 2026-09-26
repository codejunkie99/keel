//! Keel starts locally and reuses installed coding tools.

mod computer_use;
mod daemon;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "keel", about = "Local coding workspace with Laya decisions")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the local engine without a UI.
    Headless,
    /// Show local workspace and engine status.
    Status,
    /// Install or inspect the local Laya decision model.
    Laya {
        #[command(subcommand)]
        command: LayaCommand,
    },
    /// Select one prepared, low-risk native computer-use action.
    ComputerUse {
        #[command(subcommand)]
        command: ComputerUseCommand,
    },
    /// Export or summarize recorded decision receipts (read-only).
    Decisions {
        #[command(subcommand)]
        command: DecisionsCommand,
    },
    /// Manage `keel headless` as a background service.
    Daemon {
        #[command(subcommand)]
        command: DaemonCommand,
    },
}

#[derive(Subcommand)]
enum LayaCommand {
    /// Download, verify, and select the pinned local model (about 680 MB).
    Install,
    /// Show whether Laya can make decisions on this Mac.
    Status,
}

#[derive(Subcommand)]
enum ComputerUseCommand {
    /// Read one redacted decision request from stdin; print selected ID or abstention as JSON.
    Decide {
        #[arg(long, value_enum, default_value_t = computer_use::ComputerUseMode::Laya)]
        mode: computer_use::ComputerUseMode,
    },
}

#[derive(Subcommand)]
enum DecisionsCommand {
    /// Write one JSON replay case per decision receipt (JSONL) to stdout or a file.
    Export {
        /// Output file; stdout when omitted.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    /// Print counts by backend, stage, validation, fallback, and confidence.
    Report {
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum DaemonCommand {
    /// Install, enable, and start the service (captures KEEL_* env).
    Install,
    /// Stop and remove the service.
    Uninstall,
    /// Start the installed service.
    Start,
    /// Stop the service.
    Stop,
    /// Restart the service.
    Restart,
    /// Show the service manager's view of the daemon.
    Status,
}

/// No Keel cloud service is bundled. This local default cannot enable sync.
const DEFAULT_EDGE_URL: &str = "http://127.0.0.1:8787";

fn edge_url_from_env() -> String {
    std::env::var("KEEL_EDGE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_EDGE_URL.into())
}

/// mimalloc: system malloc (macOS libmalloc especially) never returns the
/// streaming churn's high-water pages, so transient allocation became
/// permanent RSS (docs/memory-plan.md §1).
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    // Long-running modes log at info, one-shot CLI commands at warn (RUST_LOG
    // overrides either).
    // loro's internal block-encode diagnostics log at info and flood
    // journald on every snapshot export — enough to fill a disk on a
    // long-running headless host. Quiet them by default (RUST_LOG still
    // overrides the whole filter).
    let long_running = matches!(&cli.command, None | Some(Command::Headless));
    let default_filter = if long_running {
        "info,loro_internal=warn,loro=warn"
    } else {
        "warn"
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| default_filter.into());
    // Long-running modes mirror stdout logging to {data_dir}/logs — a headed
    // app launched from Finder has no visible stdout, which left every sync
    // wedge report ("stale until restart") with zero diagnostics even though
    // the engine logs the exact failure line. One file per launch, previous
    // launch kept as `.old`.
    let log_file = if long_running {
        let mode = if cli.command.is_some() {
            "headless"
        } else {
            "headed"
        };
        open_log_file(mode)
    } else {
        None
    };
    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        let registry = tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer());
        match log_file {
            Some(file) => registry
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_ansi(false)
                        .with_writer(std::sync::Arc::new(file)),
                )
                .init(),
            None => registry.init(),
        }
    }

    match cli.command {
        Some(Command::ComputerUse { command }) => match command {
            ComputerUseCommand::Decide { mode } => computer_use::decide_cli(mode),
        },
        Some(Command::Headless) => {
            let runtime = tokio::runtime::Runtime::new()?;
            runtime.block_on(async {
                let engine = keel_engine::Engine::new(engine_config_from_env());
                engine.run().await
            })
        }
        Some(Command::Status) => {
            let config = engine_config_from_env();
            println!("Keel: local workspace at {}", config.data_dir.display());
            println!("Default coding tool: {:?}", config.default_harness);
            print_decision_status(&config.data_dir);
            Ok(())
        }
        Some(Command::Laya { command }) => match command {
            LayaCommand::Install => {
                let data_dir = engine_config_from_env().data_dir;
                let runtime = tokio::runtime::Runtime::new()?;
                runtime.block_on(install_laya_from_terminal(&data_dir))
            }
            LayaCommand::Status => {
                let data_dir = engine_config_from_env().data_dir;
                print_decision_status(&data_dir);
                Ok(())
            }
        },
        Some(Command::Decisions { command }) => decisions_cli(command),
        Some(Command::Daemon { command }) => match command {
            DaemonCommand::Install => daemon::install(&engine_config_from_env().data_dir),
            DaemonCommand::Uninstall => daemon::uninstall(),
            DaemonCommand::Start => daemon::start(),
            DaemonCommand::Stop => daemon::stop(),
            DaemonCommand::Restart => daemon::restart(),
            DaemonCommand::Status => daemon::status(),
        },
        None => {
            // Headed: the UI probes KEEL_IPC_PORT and connects to a running
            // daemon, or embeds the engine in-process (ARCHITECTURE §1).
            keel_ui::run_app(keel_ui::UiConfig {
                data_dir: std::env::var_os("KEEL_DATA_DIR")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(dirs_data_dir),
                ipc_port: std::env::var("KEEL_IPC_PORT")
                    .ok()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(27655),
                edge_url: edge_url_from_env(),
                workos_client_id: None,
                edge_token: None,
                org_id: None,
                default_harness: keel_ui::HarnessId::ClaudeCode,
            });
            Ok(())
        }
    }
}

fn decisions_cli(command: DecisionsCommand) -> anyhow::Result<()> {
    use keel_engine::decision_log;
    use std::io::Write;

    // ponytail: local profile only; add a --profile flag when synced profiles need it.
    let store_root = engine_config_from_env()
        .data_dir
        .join("profiles")
        .join("local");
    let decisions = decision_log::read_decisions(&store_root)?;
    match command {
        DecisionsCommand::Export { out } => {
            let mut writer: Box<dyn Write> = match &out {
                Some(path) => Box::new(std::io::BufWriter::new(std::fs::File::create(path)?)),
                None => Box::new(std::io::stdout().lock()),
            };
            for decision in &decisions {
                serde_json::to_writer(&mut writer, &decision.replay_case())?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
            if let Some(path) = out {
                eprintln!(
                    "Wrote {} replay cases to {}",
                    decisions.len(),
                    path.display()
                );
            }
        }
        DecisionsCommand::Report { json } => {
            let report = decision_log::report(&decisions);
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_decision_report(&report);
            }
        }
    }
    Ok(())
}

fn print_decision_report(report: &keel_engine::decision_log::DecisionReport) {
    if report.total == 0 {
        println!("No decision receipts recorded yet.");
        return;
    }
    let pct = |n: usize| 100.0 * n as f64 / report.total as f64;
    println!(
        "Decision receipts: {} across {} chats",
        report.total, report.chats
    );
    println!(
        "  selected   {:>5}  ({:.0}%)",
        report.selected,
        pct(report.selected)
    );
    println!(
        "  abstained  {:>5}  ({:.0}%)",
        report.abstained,
        pct(report.abstained)
    );
    println!(
        "  fallback   {:>5}  ({:.0}%)",
        report.with_fallback,
        pct(report.with_fallback)
    );
    println!(
        "  outcome    {:>5}  ({:.0}%)",
        report.with_outcome,
        pct(report.with_outcome)
    );
    for (label, counts) in [
        ("backend", &report.by_backend),
        ("stage", &report.by_stage),
        ("validation", &report.by_validation),
    ] {
        let parts: Vec<String> = counts.iter().map(|(k, v)| format!("{k} {v}")).collect();
        println!("{label:<11}{}", parts.join(", "));
    }
    match report.mean_confidence {
        Some(mean) => println!("mean confidence {mean:.2}"),
        None => println!("mean confidence not reported"),
    }
}

fn print_decision_status(data_dir: &std::path::Path) {
    use keel_engine::decision_mode::{self, DecisionMode};

    let selected = decision_mode::read(data_dir);
    let laya = keel_ui::onboarding::laya_status(data_dir);
    let jev_ready = decision_mode::protected_typesafe_key_path().is_some();
    let effective = match selected {
        DecisionMode::Laya if laya.installed => "Laya (local)",
        DecisionMode::Jev if jev_ready => "Jev (direct TypeSafe)",
        _ => "Normal harness",
    };
    println!("Selected decision backend: {selected:?}");
    println!("Available intake backend (local checks): {effective}");
    println!("Laya: {}", laya.message);
    println!(
        "Jev: {}",
        if jev_ready {
            "protected local credential available"
        } else {
            "protected local credential unavailable"
        }
    );
    println!("External tools: the backend chooses once at eligible new-task intake.");
    println!("Embedded DeepSeek: the backend can also choose a bounded step focus.");
    println!("The task chat shows live selection only while it actually runs, then its result.");
}

async fn install_laya_from_terminal(data_dir: &std::path::Path) -> anyhow::Result<()> {
    use keel_engine::decision_mode::{self, LayaInstallProgress};
    use std::io::Write;

    let status = keel_ui::onboarding::laya_status(data_dir);
    if !status.installed && !status.downloadable {
        anyhow::bail!("Laya cannot be installed here: {}", status.message);
    }
    if status.installed {
        println!("Laya model is already available; no download needed.");
    } else {
        println!(
            "Downloading the pinned Laya model to {}",
            data_dir.join("laya/model").display()
        );
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        let install_dir = data_dir.to_path_buf();
        let install = tokio::spawn(async move {
            decision_mode::install_laya_model_with_progress(&install_dir, progress_tx).await
        });
        let mut last_percent = None;
        let mut verifying = false;
        while let Some(progress) = progress_rx.recv().await {
            match progress {
                LayaInstallProgress::Downloading {
                    downloaded_bytes,
                    total_bytes,
                } if total_bytes > 0 => {
                    let percent = downloaded_bytes.saturating_mul(100) / total_bytes;
                    if last_percent != Some(percent) {
                        let filled = (percent.min(100) / 5) as usize;
                        print!(
                            "\rLaya [{}{}] {percent:>3}% ({}/{} MB)",
                            "#".repeat(filled),
                            "-".repeat(20 - filled),
                            downloaded_bytes / 1_000_000,
                            total_bytes / 1_000_000
                        );
                        std::io::stdout().flush()?;
                        last_percent = Some(percent);
                    }
                }
                LayaInstallProgress::Verifying => {
                    println!("\nDownload complete. Verifying model files…");
                    verifying = true;
                }
                _ => {}
            }
        }
        if last_percent.is_some() && !verifying {
            println!();
        }
        install.await??;
        if decision_mode::laya_assets_in(data_dir).is_none() {
            anyhow::bail!("Laya downloaded, but the worker or model is not ready for decisions");
        }
        println!("Laya model verified and installed.");
    }
    keel_ui::onboarding::save_mode(data_dir, keel_ui::onboarding::DecisionMode::Laya)?;
    println!(
        "Laya selected for eligible new tasks. The chat will show when it actually makes a decision."
    );
    Ok(())
}

/// The headless engine is local-only, even if stale cloud environment values exist.
fn engine_config_from_env() -> keel_engine::EngineConfig {
    keel_engine::EngineConfig {
        data_dir: std::env::var_os("KEEL_DATA_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(dirs_data_dir),
        edge_url: edge_url_from_env(),
        ipc_port: std::env::var("KEEL_IPC_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(27655),
        default_harness: harness_from_env(),
        org_id: None,
        workos_client_id: None,
        edge_token: None,
    }
}

/// `KEEL_HARNESS` (kebab-case id) picks the default harness for chats without a
/// config row — `mock` powers the e2e smoke; default `claude-code`.
fn harness_from_env() -> keel_engine::HarnessId {
    match std::env::var("KEEL_HARNESS").as_deref().map(str::trim) {
        Ok("mock") => keel_engine::HarnessId::Mock,
        Ok("codex") => keel_engine::HarnessId::Codex,
        Ok("cursor") => keel_engine::HarnessId::Cursor,
        Ok("grok") => keel_engine::HarnessId::Grok,
        Ok("hermes") => keel_engine::HarnessId::Hermes,
        Ok("pi") => keel_engine::HarnessId::Pi,
        Ok("dsh") => keel_engine::HarnessId::Dsh,
        Ok("og") => keel_engine::HarnessId::Og,
        _ => keel_engine::HarnessId::ClaudeCode,
    }
}

fn dirs_data_dir() -> std::path::PathBuf {
    let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME not set"));
    home.join(".keel")
}

fn open_log_file(mode: &str) -> Option<std::fs::File> {
    let dir = std::env::var_os("KEEL_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(dirs_data_dir)
        .join("logs");
    open_log_file_in(&dir, mode)
}

/// Dir-parameterized body of [`open_log_file`] (unit-testable without env).
fn open_log_file_in(dir: &std::path::Path, mode: &str) -> Option<std::fs::File> {
    std::fs::create_dir_all(dir).ok()?;
    let path = dir.join(format!("keel-{mode}.log"));
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        // Probe the CURRENT inode for a live writer before touching it.
        let preexisting = path.exists();
        let existing = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .ok()?;
        let rc = unsafe { libc::flock(existing.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            // A live process owns the canonical log — leave it alone.
            return std::fs::File::create(
                dir.join(format!("keel-{mode}.{}.log", std::process::id())),
            )
            .ok();
        }
        // No live writer: rotate, create fresh, and lock it as ours. (The
        // probe's flock dies with `existing`; a first-ever launch has nothing
        // to rotate — the probe itself created the empty file.)
        drop(existing);
        if preexisting {
            let _ = std::fs::rename(&path, dir.join(format!("keel-{mode}.log.old")));
        }
        let file = std::fs::File::create(&path).ok()?;
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        sweep_stale_pid_logs(dir, mode);
        Some(file)
    }
    #[cfg(not(unix))]
    {
        let _ = std::fs::rename(&path, dir.join(format!("keel-{mode}.log.old")));
        std::fs::File::create(&path).ok()
    }
}

#[cfg(all(test, unix))]
mod log_file_tests {
    use super::open_log_file_in;

    #[test]
    fn second_launch_never_rotates_a_live_processes_log() {
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        // First launch owns the canonical file and keeps writing.
        let first = open_log_file_in(dir, "headed").expect("first log");
        assert!(dir.join("keel-headed.log").is_file());
        // Second launch while the first is alive: canonical file untouched,
        // pid-suffixed overflow file instead (the 2026-08-04 clobber).
        let second = open_log_file_in(dir, "headed").expect("second log");
        let pid_path = dir.join(format!("keel-headed.{}.log", std::process::id()));
        assert!(pid_path.is_file(), "expected pid-suffixed overflow log");
        assert!(
            !dir.join("keel-headed.log.old").exists(),
            "live canonical log must not be rotated away"
        );
        drop(second);
        // After the owner exits, a fresh launch rotates normally.
        drop(first);
        let third = open_log_file_in(dir, "headed").expect("third log");
        assert!(
            dir.join("keel-headed.log.old").is_file(),
            "rotation resumes"
        );
        drop(third);
    }
}

/// Delete `keel-{mode}.{pid}.log` overflow files older than a week — they
/// only exist when a second instance raced a live one for the canonical log.
#[cfg(unix)]
fn sweep_stale_pid_logs(dir: &std::path::Path, mode: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let prefix = format!("keel-{mode}.");
    let week = std::time::Duration::from_secs(7 * 24 * 60 * 60);
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(middle) = name
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix(".log"))
        else {
            continue;
        };
        if !middle.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > week);
        if stale {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

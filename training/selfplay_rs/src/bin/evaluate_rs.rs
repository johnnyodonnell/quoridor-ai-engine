//! evaluate_rs: the Rust port of `training/evaluate.py` (port of
//! chess-ai-engine's selfplay_rs/src/bin/evaluate_rs.rs).
//!
//! Plays a candidate snapshot against the active pool, refits Elo, updates
//! pool.json, and promotes the leader to best.onnx. The orchestrator launches
//! this after each snapshot.
//!
//!   evaluate_rs --run-dir <dir> --candidate <snap>.safetensors [flags]
//!   evaluate_rs --run-dir <dir> --fit-only          (diagnostic: refit Elo, print)
//!   evaluate_rs --match-only <a> <b> [flags]        (diagnostic: one pairing's W/D/L;
//!                                                    <a>/<b> = .safetensors or "random")

use std::path::PathBuf;
use std::process::exit;

/// Read a `--key value` flag, falling back to `default`.
fn flag(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn require(args: &[String], key: &str) -> String {
    let v = flag(args, key, "");
    if v.is_empty() {
        eprintln!("missing required {key}");
        eprintln!("usage: evaluate_rs --run-dir <dir> --candidate <snap>.safetensors \\");
        eprintln!("  [--games-per-pair 100] [--sims 200] [--opening-plies 10] \\");
        eprintln!("  [--n-top 2] [--n-anchors 3] [--serve-margin 35] [--seed 0]");
        exit(2);
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games_per_pair: usize = flag(&args, "--games-per-pair", "100").parse().unwrap();
    let sims: usize = flag(&args, "--sims", "200").parse().unwrap();
    let opening_plies: usize = flag(&args, "--opening-plies", "10").parse().unwrap();
    let seed: u64 = flag(&args, "--seed", "0").parse().unwrap();

    if let Some(i) = args.iter().position(|a| a == "--match-only") {
        let (a, b) = match (args.get(i + 1), args.get(i + 2)) {
            (Some(a), Some(b)) => (a.clone(), b.clone()),
            _ => {
                eprintln!("usage: evaluate_rs --match-only <a> <b> [--games-per-pair N] [--sims N] [--opening-plies N] [--seed N]");
                exit(2);
            }
        };
        selfplay_rs::eval::match_only(&a, &b, games_per_pair, sims, opening_plies, seed);
        return;
    }

    let run_dir = PathBuf::from(require(&args, "--run-dir"));
    if args.iter().any(|a| a == "--fit-only") {
        selfplay_rs::eval::fit_only(&run_dir);
        return;
    }
    let candidate = PathBuf::from(require(&args, "--candidate"));
    selfplay_rs::eval::evaluate(
        &run_dir,
        &candidate,
        games_per_pair,
        sims,
        opening_plies,
        flag(&args, "--n-top", "2").parse().unwrap(),
        flag(&args, "--n-anchors", "3").parse().unwrap(),
        flag(&args, "--serve-margin", "35").parse().unwrap(),
        seed,
    );
}

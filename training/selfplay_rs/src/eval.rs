//! Rust port of `training/evaluate.py`: self-play evaluation, Elo rating, and
//! model serving (port of chess-ai-engine's `selfplay_rs/src/eval.rs`).
//!
//! A candidate snapshot plays match games against an active pool (top
//! performers + a fixed random anchor + frozen ex-snapshot anchors spread across
//! the strength range), a lightweight Bradley-Terry Elo is refit over all
//! accumulated results (random pinned at 0), the pool is updated, and the
//! highest-rated model is promoted to `<run-dir>/best.onnx` if it clears the
//! served model by a confidence margin.
//!
//! Match play reproduces the old pyo3 `MatchEngine` (quoridor_rs/src/matches.rs)
//! step for step: lockstep batched MCTS across all games of a pairing, a fresh
//! tree per move, no root noise, visit-proportional sampling for the first
//! `opening_plies` plies and the most visited move after. Games advance and
//! consume the RNG in the same order, and the forward is the fp32 explicit-BN
//! net (bit-exact to PyTorch), so a pairing with the same seed reproduces the
//! Python evaluator's results. Eval is low throughput, so this is a plain
//! synchronous loop with none of the self-play worker's AOTI/pipeline machinery.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use indexmap::IndexMap;
use quoridor_core::board::{Board, ENC_LEN, INPUT_CHANNELS, POLICY_SIZE, SIZE};
use quoridor_core::mcts::{backprop, expand_node, sample_move, walk_to_leaf, Node};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

use crate::net::Net;

const RANDOM: &str = "random"; // reserved name for the fixed floor anchor (Elo pinned at 0)

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// A Python-style list repr (`['a', 'b']`), so eval.log lines read the same as
/// they did under evaluate.py.
fn py_list(xs: &[String]) -> String {
    let items: Vec<String> = xs.iter().map(|x| format!("'{x}'")).collect();
    format!("[{}]", items.join(", "))
}

// ---------------------------------------------------------------------------
// Players
// ---------------------------------------------------------------------------
enum Player {
    Random,
    Net(Net),
}

impl Player {
    fn is_random(&self) -> bool {
        matches!(self, Player::Random)
    }
}

/// Run `net` (fp32) over `m` encodings (`enc` is `m * ENC_LEN` f32) and return
/// `(logits[m*POLICY_SIZE], values[m])` on the host.
fn eval_batch(net: &Net, dev: Device, enc: &[f32], m: usize) -> (Vec<f32>, Vec<f32>) {
    let x = Tensor::from_slice(enc)
        .reshape([m as i64, INPUT_CHANNELS as i64, SIZE as i64, SIZE as i64])
        .to_device(dev)
        .to_kind(Kind::Float);
    let (logits, values) = tch::no_grad(|| net.forward_unfolded(&x));
    let lc = logits.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
    let vc = values.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
    let mut lv = vec![0f32; m * POLICY_SIZE];
    lc.copy_data(&mut lv, m * POLICY_SIZE);
    let mut vv = vec![0f32; m];
    vc.copy_data(&mut vv, m);
    (lv, vv)
}

// ---------------------------------------------------------------------------
// Match play (port of quoridor_rs MatchEngine)
// ---------------------------------------------------------------------------
struct MatchGame {
    board: Board,
    a_first: bool, // A plays player 0
    arena: Vec<Node>,
    sim: usize,
    pending: Option<(Vec<usize>, Board)>, // (path, leaf board) awaiting an eval
    result: Option<i8>,                   // player 0 POV once over
}

impl MatchGame {
    /// 0 if A is to move, 1 if B.
    fn mover(&self) -> usize {
        if (self.board.turn() == 0) == self.a_first {
            0
        } else {
            1
        }
    }

    fn play(&mut self, idx: u16) {
        self.board.push(idx);
        self.arena = vec![Node::new(0.0, -1)];
        self.sim = 0;
        self.result = self.board.outcome();
    }
}

/// Advance one game until it needs an NN eval (staged in `pending`) or ends.
fn advance(
    g: &mut MatchGame,
    sims: usize,
    opening_plies: usize,
    random: [bool; 2],
    rng: &mut StdRng,
) {
    while g.result.is_none() && g.pending.is_none() {
        if random[g.mover()] {
            let moves = g.board.legal_moves();
            let idx = moves[rng.gen_range(0..moves.len())];
            g.play(idx);
            continue;
        }
        if !g.arena[0].expanded {
            g.pending = Some((vec![0], g.board.clone()));
            break;
        }
        if g.sim >= sims {
            let temp = if g.board.ply() < opening_plies { 1.0 } else { 0.0 };
            let idx = sample_move(&g.arena, 0, temp, rng);
            g.play(idx);
            continue;
        }
        let mut b = g.board.clone();
        let path = walk_to_leaf(&g.arena, 0, &mut b);
        match b.terminal_value() {
            Some(t) => {
                backprop(&mut g.arena, &path, t as f64);
                g.sim += 1;
            }
            None => g.pending = Some((path, b)),
        }
    }
}

/// Play `n_games` between two players, alternating who moves first. Net
/// evaluations are batched across all games where the same player needs one.
/// Returns (wins, draws, losses) from `player_a`'s perspective.
fn play_match(
    player_a: &Player,
    player_b: &Player,
    n_games: usize,
    sims: usize,
    opening_plies: usize,
    seed: u64,
    dev: Device,
) -> (i32, i32, i32) {
    let players = [player_a, player_b];
    let random = [player_a.is_random(), player_b.is_random()];
    let sims = sims.max(1);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut games: Vec<MatchGame> = (0..n_games)
        .map(|i| MatchGame {
            board: Board::new(),
            a_first: i % 2 == 0,
            arena: vec![Node::new(0.0, -1)],
            sim: 0,
            pending: None,
            result: None,
        })
        .collect();

    loop {
        // Advance every unfinished game to its next leaf (in game order).
        let mut enc: Vec<f32> = Vec::new();
        let mut who: Vec<usize> = Vec::new();
        for g in games.iter_mut() {
            advance(g, sims, opening_plies, random, &mut rng);
            if let Some((_, b)) = &g.pending {
                let start = enc.len();
                enc.resize(start + ENC_LEN, 0.0);
                b.encode_into(&mut enc[start..]);
                who.push(g.mover());
            }
        }
        let m = who.len();
        if m == 0 {
            break;
        }

        // Run each player's net over its own rows (row order preserved).
        let mut logits = vec![0f32; m * POLICY_SIZE];
        let mut values = vec![0f32; m];
        for k in 0..2 {
            let rows: Vec<usize> = (0..m).filter(|&r| who[r] == k).collect();
            if rows.is_empty() {
                continue;
            }
            let net = match players[k] {
                Player::Net(n) => n,
                Player::Random => unreachable!("random player never needs the net"),
            };
            let mut sub = Vec::with_capacity(rows.len() * ENC_LEN);
            for &r in &rows {
                sub.extend_from_slice(&enc[r * ENC_LEN..(r + 1) * ENC_LEN]);
            }
            let (l, v) = eval_batch(net, dev, &sub, rows.len());
            for (j, &r) in rows.iter().enumerate() {
                logits[r * POLICY_SIZE..(r + 1) * POLICY_SIZE]
                    .copy_from_slice(&l[j * POLICY_SIZE..(j + 1) * POLICY_SIZE]);
                values[r] = v[j];
            }
        }

        // Apply results in the same game order the leaves were collected.
        let mut r = 0;
        for g in games.iter_mut() {
            let Some((path, b)) = g.pending.take() else { continue };
            let leaf = *path.last().unwrap();
            expand_node(&mut g.arena, leaf, &b, &logits[r * POLICY_SIZE..(r + 1) * POLICY_SIZE]);
            if path.len() > 1 {
                backprop(&mut g.arena, &path, values[r] as f64);
                g.sim += 1;
            }
            r += 1;
        }
    }

    let (mut wins, mut draws, mut losses) = (0, 0, 0);
    for g in &games {
        let r0 = g.result.expect("match ended with an unfinished game");
        let a = if g.a_first { r0 } else { -r0 };
        if a > 0 {
            wins += 1;
        } else if a < 0 {
            losses += 1;
        } else {
            draws += 1;
        }
    }
    (wins, draws, losses)
}

// ---------------------------------------------------------------------------
// Elo fit (Bradley-Terry, coordinate-Newton; random pinned, L2-regularized)
// ---------------------------------------------------------------------------
/// `games`: (a, b, score_a, n): in n games between a and b, a scored score_a
/// points (win=1, draw=0.5). Names in `fixed` are held at their fixed rating to
/// pin the scale; `reg` pulls ratings toward 0 so a 100% sweep stays finite.
fn fit_elo(
    names: &[String],
    games: &[(String, String, f64, i64)],
    fixed: &HashMap<String, f64>,
    reg: f64,
    iters: usize,
) -> HashMap<String, f64> {
    let mut r: HashMap<String, f64> =
        names.iter().map(|n| (n.clone(), *fixed.get(n).unwrap_or(&0.0))).collect();
    let q = 10f64.ln() / 400.0;

    // adjacency: name -> Vec<(opponent, score_for_name, n)>
    let mut adj: HashMap<String, Vec<(String, f64, f64)>> =
        names.iter().map(|n| (n.clone(), Vec::new())).collect();
    for (a, b, score_a, n) in games {
        if *n <= 0 {
            continue;
        }
        let nf = *n as f64;
        adj.get_mut(a).unwrap().push((b.clone(), *score_a, nf));
        adj.get_mut(b).unwrap().push((a.clone(), nf - *score_a, nf));
    }

    for _ in 0..iters {
        for p in names {
            if fixed.contains_key(p) {
                continue;
            }
            let mut g = 0.0;
            let mut h = 0.0;
            let rp = r[p];
            for (opp, score_p, n) in &adj[p] {
                let e = 1.0 / (1.0 + 10f64.powf((r[opp] - rp) / 400.0));
                g += q * (score_p - n * e);
                h += q * q * n * e * (1.0 - e);
            }
            g -= reg * rp;
            h += reg;
            if h > 1e-12 {
                *r.get_mut(p).unwrap() += g / h;
            }
        }
    }
    for (n, v) in fixed {
        r.insert(n.clone(), *v);
    }
    r
}

// ---------------------------------------------------------------------------
// Pool (on-disk JSON, schema-compatible with evaluate.py except pt -> st)
// ---------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Clone, Default)]
struct ModelEntry {
    #[serde(default)]
    st: Option<String>, // safetensors path relative to run-dir (was "pt")
    #[serde(default)]
    onnx: Option<String>,
    #[serde(default)]
    rating: Option<f64>,
}

#[derive(Serialize, Deserialize, Clone)]
struct MatchResult {
    a: String,
    b: String,
    score_a: f64,
    n: i64,
}

#[derive(Serialize, Deserialize, Default)]
struct Pool {
    #[serde(default)]
    served: Option<String>,
    #[serde(default)]
    served_rating: Option<f64>,
    #[serde(default)]
    models: IndexMap<String, ModelEntry>,
    #[serde(default)]
    results: Vec<MatchResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    top: Vec<String>,
}

fn load_pool(path: &Path) -> Pool {
    if path.exists() {
        let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
    } else {
        Pool::default()
    }
}

fn save_pool(path: &Path, pool: &Pool) {
    let tmp = path.with_extension("tmp");
    let text = serde_json::to_string_pretty(pool).expect("serialize pool");
    fs::write(&tmp, text).unwrap_or_else(|e| panic!("write {tmp:?}: {e}"));
    fs::rename(&tmp, path).unwrap_or_else(|e| panic!("rename {tmp:?} -> {path:?}: {e}"));
}

/// Active opponents = top `n_top` rated + random + `n_anchors` frozen snapshots
/// whose ratings most evenly cover (0, rating of the n_top-th). Returns
/// (active opponent set, top list). `models` must be in pool insertion order.
fn select_anchors(
    ratings: &HashMap<String, f64>,
    models: &[String],
    n_top: usize,
    n_anchors: usize,
) -> (Vec<String>, Vec<String>) {
    let rget = |m: &str| ratings.get(m).copied().unwrap_or(0.0);

    let mut ranked: Vec<String> = models.iter().filter(|m| *m != RANDOM).cloned().collect();
    // Stable descending sort by rating (ties keep insertion order, like Python).
    ranked.sort_by(|a, b| rget(b).partial_cmp(&rget(a)).unwrap());

    let top: Vec<String> = ranked.iter().take(n_top).cloned().collect();
    let mut active = top.clone();
    active.push(RANDOM.to_string());

    let below: Vec<String> = ranked.iter().skip(n_top).cloned().collect();
    if !below.is_empty() {
        let ceiling = if let Some(last_top) = top.last() {
            rget(last_top)
        } else {
            below.iter().map(|m| rget(m)).fold(f64::NEG_INFINITY, f64::max)
        };
        let k = n_anchors.min(below.len());
        let mut pool: Vec<String> = below;
        for i in 0..k {
            let t = ceiling * (i + 1) as f64 / (k + 1) as f64;
            // Closest unused model (first on ties, matching Python min()).
            let bi = pool
                .iter()
                .enumerate()
                .min_by(|(_, m1), (_, m2)| {
                    (rget(m1) - t).abs().partial_cmp(&(rget(m2) - t).abs()).unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap();
            active.push(pool.remove(bi));
        }
    }
    (active, top)
}

/// Copy the leader's onnx to best.onnx. Every snapshot ships an .onnx and random
/// is never a leader, so there is no export fallback. best.pt is no longer
/// written (the browser serves best.onnx only), as in chess-ai-engine.
fn serve(run_dir: &Path, pool: &Pool, name: &str) {
    let onnx_rel = pool.models[name]
        .onnx
        .as_ref()
        .unwrap_or_else(|| panic!("model {name} has no onnx to serve"));
    let onnx = run_dir.join(onnx_rel);
    fs::copy(&onnx, run_dir.join("best.onnx"))
        .unwrap_or_else(|e| panic!("copy {onnx:?} -> best.onnx: {e}"));
}

fn load_player(st_path: &Path, dev: Device) -> Player {
    Player::Net(Net::load(st_path.to_str().expect("non-utf8 path"), dev, Kind::Float))
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    run_dir: &Path,
    candidate: &Path,
    games_per_pair: usize,
    sims: usize,
    opening_plies: usize,
    n_top: usize,
    n_anchors: usize,
    serve_margin: f64,
    seed: u64,
) {
    let t0 = Instant::now();
    let run_dir = run_dir.canonicalize().unwrap_or_else(|e| panic!("run dir {run_dir:?}: {e}"));
    let candidate = candidate.canonicalize().unwrap_or_else(|e| panic!("candidate {candidate:?}: {e}"));
    let pool_path = run_dir.join("pool.json");
    let mut pool = load_pool(&pool_path);
    let dev = Device::cuda_if_available();

    let cand_name = candidate
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("candidate stem")
        .to_string();
    let cand_onnx = candidate.with_extension("onnx");
    let dev_name = if matches!(dev, Device::Cuda(_)) { "cuda" } else { "cpu" };
    println!("[eval] candidate={cand_name} device={dev_name}");

    // Register candidate (paths stored relative to run-dir for portability).
    let rel = |p: &Path| -> String {
        p.strip_prefix(&run_dir)
            .map(|r| r.to_string_lossy().into_owned())
            .unwrap_or_else(|_| p.to_string_lossy().into_owned())
    };
    pool.models.insert(
        cand_name.clone(),
        ModelEntry { st: Some(rel(&candidate)), onnx: Some(rel(&cand_onnx)), rating: None },
    );
    pool.models
        .entry(RANDOM.to_string())
        .or_insert(ModelEntry { st: None, onnx: None, rating: Some(0.0) });

    // Pick the active opponent set from prior ratings (top-N + random + anchors).
    let prior: HashMap<String, f64> =
        pool.models.iter().map(|(m, e)| (m.clone(), e.rating.unwrap_or(0.0))).collect();
    let model_names: Vec<String> = pool.models.keys().cloned().collect();
    let (active, _) = select_anchors(&prior, &model_names, n_top, n_anchors);
    let mut opponents: Vec<String> = active.into_iter().filter(|m| *m != cand_name).collect();
    if opponents.is_empty() {
        opponents = vec![RANDOM.to_string()];
    }
    println!("[eval] opponents={}", py_list(&opponents));

    let cand_player = load_player(&candidate, dev);

    // Resolve each opponent's safetensors path up front, so the match loop can
    // mutate pool.results without holding a borrow on the pool.
    let opp_specs: Vec<(String, Option<String>)> = opponents
        .iter()
        .map(|o| {
            let st = if o == RANDOM {
                None
            } else {
                Some(pool.models[o].st.clone().unwrap_or_else(|| panic!("model {o} has no st path")))
            };
            (o.clone(), st)
        })
        .collect();

    let mut cand_vs_random: Option<f64> = None;
    for (i, (opp, st)) in opp_specs.iter().enumerate() {
        let opp_player = match st {
            None => Player::Random,
            Some(p) => load_player(&run_dir.join(p), dev),
        };
        let t = Instant::now();
        let (w, d, l) = play_match(
            &cand_player,
            &opp_player,
            games_per_pair,
            sims,
            opening_plies,
            seed + i as u64,
            dev,
        );
        let score_a = w as f64 + 0.5 * d as f64;
        pool.results.push(MatchResult {
            a: cand_name.clone(),
            b: opp.clone(),
            score_a,
            n: (w + d + l) as i64,
        });
        if opp == RANDOM {
            cand_vs_random = Some(score_a / (w + d + l).max(1) as f64);
        }
        println!("[eval] {cand_name} vs {opp}: W{w} D{d} L{l} ({:.0}s)", t.elapsed().as_secs_f64());
    }

    // Refit Elo over all accumulated results.
    let names: Vec<String> = pool.models.keys().cloned().collect();
    let games: Vec<(String, String, f64, i64)> =
        pool.results.iter().map(|r| (r.a.clone(), r.b.clone(), r.score_a, r.n)).collect();
    let fixed: HashMap<String, f64> = HashMap::from([(RANDOM.to_string(), 0.0)]);
    let ratings = fit_elo(&names, &games, &fixed, 1e-4, 400);
    for m in &names {
        pool.models.get_mut(m).unwrap().rating = Some(round1(ratings[m]));
    }

    // Recompute the active set for the next round and decide serving.
    let (_, top) = select_anchors(&ratings, &names, n_top, n_anchors);
    let leader = top.first().cloned();
    pool.top = top;

    let mut ranked_names = names.clone();
    ranked_names.sort_by(|a, b| {
        ratings.get(b).unwrap_or(&0.0).partial_cmp(ratings.get(a).unwrap_or(&0.0)).unwrap()
    });
    let rating_str = ranked_names
        .iter()
        // {:?} prints 0.0 / 551.6 like Python's float repr ({} would print "0").
        .map(|m| format!("{m}={:?}", pool.models[m].rating.unwrap_or(0.0)))
        .collect::<Vec<_>>()
        .join(", ");
    println!("[eval] ratings: {rating_str}");

    let served = pool.served.clone();
    // Compare against the served model's CURRENT refit rating, not the frozen
    // served_rating: every refit drifts the whole scale, so a stale threshold
    // becomes unbeatable and best.onnx would stick forever.
    let served_now = served.as_ref().and_then(|s| ratings.get(s).copied());
    let sane = cand_vs_random.map_or(true, |c| c >= 0.6);
    if let Some(leader) = leader.filter(|_| sane) {
        let lead_r = ratings[&leader];
        if served.is_none() || lead_r > served_now.unwrap_or(-1e9) + serve_margin {
            serve(&run_dir, &pool, &leader);
            pool.served = Some(leader.clone());
            pool.served_rating = Some(round1(lead_r));
            println!("[eval] SERVED {leader} (rating={lead_r:.1})");
        } else {
            println!(
                "[eval] kept {} (rating={:.1}); leader {leader}={lead_r:.1} within margin",
                served.as_deref().unwrap_or("?"),
                served_now.unwrap_or(0.0)
            );
        }
    } else if !sane {
        println!(
            "[eval] candidate failed random sanity gate (score vs random={:.2}); not serving",
            cand_vs_random.unwrap_or(0.0)
        );
    }

    save_pool(&pool_path, &pool);
    println!("[eval] done in {:.0}s", t0.elapsed().as_secs_f64());
}

/// Diagnostic hook: load pool.json, refit Elo over the recorded results, and
/// print `name\trating` lines sorted by rating (no match play). Used to check
/// fit_elo parity against the Python reference.
pub fn fit_only(run_dir: &Path) {
    let pool = load_pool(&run_dir.join("pool.json"));
    let names: Vec<String> = pool.models.keys().cloned().collect();
    let games: Vec<(String, String, f64, i64)> =
        pool.results.iter().map(|r| (r.a.clone(), r.b.clone(), r.score_a, r.n)).collect();
    let fixed: HashMap<String, f64> = HashMap::from([(RANDOM.to_string(), 0.0)]);
    let ratings = fit_elo(&names, &games, &fixed, 1e-4, 400);
    let mut ns = names.clone();
    ns.sort_by(|a, b| ratings[b].partial_cmp(&ratings[a]).unwrap());
    for n in &ns {
        println!("{n}\t{:.4}", ratings[n]);
    }
}

/// Diagnostic hook: play one pairing between two snapshots (either may be
/// "random") and print W/D/L, without touching pool.json. Used to check
/// match-play parity against the Python evaluator.
pub fn match_only(a: &str, b: &str, games: usize, sims: usize, opening_plies: usize, seed: u64) {
    let dev = Device::cuda_if_available();
    let load = |s: &str| if s == RANDOM { Player::Random } else { load_player(Path::new(s), dev) };
    let (pa, pb) = (load(a), load(b));
    let (w, d, l) = play_match(&pa, &pb, games, sims, opening_plies, seed, dev);
    println!("W{w} D{d} L{l}");
}

#[cfg(test)]
mod tests {
    use super::*;

    // fit_elo: a 100% sweep stays finite (reg), and order matches strength.
    #[test]
    fn elo_orders_by_strength() {
        let names: Vec<String> = ["random", "weak", "strong"].iter().map(|s| s.to_string()).collect();
        let games = vec![
            ("weak".to_string(), "random".to_string(), 80.0, 100),
            ("strong".to_string(), "random".to_string(), 95.0, 100),
            ("strong".to_string(), "weak".to_string(), 70.0, 100),
        ];
        let fixed = HashMap::from([("random".to_string(), 0.0)]);
        let r = fit_elo(&names, &games, &fixed, 1e-4, 400);
        assert!((r["random"]).abs() < 1e-9);
        assert!(r["strong"] > r["weak"], "strong {} should beat weak {}", r["strong"], r["weak"]);
        assert!(r["weak"] > 0.0, "weak {} should beat random", r["weak"]);
    }

    // Random vs random never touches the net and every game finishes.
    #[test]
    fn random_vs_random_completes() {
        let (w, d, l) = play_match(&Player::Random, &Player::Random, 6, 8, 4, 0, Device::Cpu);
        assert_eq!(w + d + l, 6);
    }
}

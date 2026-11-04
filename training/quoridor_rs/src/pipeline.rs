//! Decoupled non-blocking self-play engine — a port of chess-ai-engine's
//! `chess_rs/src/pipeline.rs` (commit 53b1b83) with the game swapped out.
//!
//! Workers NEVER block on the GPU: a worker advances ONE game until it produces
//! exactly one NN leaf, pushes that leaf to an inference bucket, and immediately
//! moves on to another game. So CPU tree-walking overlaps the GPU forward.
//!
//! Topology:
//!   - A GLOBAL priority queue of in-flight games, keyed by ply count (most plies
//!     first), so near-complete games drain first and the working set of search
//!     trees stays bounded.
//!   - `n_threads` WORKER threads (GIL released) pop the highest-ply game, apply
//!     any pending NN result (expand + backprop), advance the per-game FSM until
//!     it yields ONE leaf eval or completes the game, then pop the next. If the
//!     queue is empty a worker SPAWNS a fresh game.
//!   - TWO inference buckets: workers fill one; when it reaches `bucket_size` it
//!     is handed to the consumer and workers fill the other. Flush-on-SIZE only.
//!     If BOTH are full all workers block (backpressure); in-flight game count
//!     self-regulates to ~2 * bucket_size.
//!   - ONE CONSUMER thread runs the Python `forward` under the GIL, attaches each
//!     result row to its game and re-queues it. It does no tree work.
//!
//! One outstanding NN leaf per game at all times (no virtual loss). A finished
//! game's rows go to the Python `submit_game(rows)` callback.

use numpy::ndarray::{Array1, Array3, Array4};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use quoridor_core::board::{Board, ENC_LEN, INPUT_CHANNELS, POLICY_SIZE, SIZE};
use quoridor_core::mcts::{
    add_dirichlet_noise, advance_subtree, backprop, expand_node, sample_move, visits_to_pi,
    walk_to_leaf, Game,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

type Row = (Vec<f32>, Vec<f32>, f32); // (state, pi, z)

// --- per-game seeding (splitmix64 mixing; determinism not required, just spread) ---
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// --- per-game FSM -----------------------------------------------------------
enum GamePhase {
    NeedRootExpand,    // next: expand root (1 eval, unless root already expanded)
    Simulating(usize), // current simulation index s in 0..sims
}

enum LeafKind {
    RootExpand, // result -> expand_node(root) + dirichlet; no backprop
    SimLeaf,    // result -> expand_node(leaf) + backprop(path, value)
}

/// Search context for the ONE outstanding leaf. `board` is the walked board at
/// the leaf; `expand_node` needs its legal moves when the result comes back.
struct LeafCtx {
    kind: LeafKind,
    path: Vec<usize>,
    board: Board,
}

struct EvalResult {
    logits: Vec<f32>, // POLICY_SIZE
    value: f32,
}

/// A game in flight. Boxed so it moves queue<->bucket<->worker by pointer.
struct InFlight {
    id: u64,
    game: Game,
    rng: StdRng,
    phase: GamePhase,
    pending: Option<EvalResult>, // written by the consumer, consumed by a worker
    leaf_ctx: Option<LeafCtx>,   // Some exactly while an eval is outstanding
    staged_enc: Vec<f32>,        // encoded leaf board, copied into the bucket on push
}

fn spawn_fresh(shared: &Shared) -> InFlight {
    let id = shared.id_counter.fetch_add(1, AtomicOrdering::Relaxed);
    let seed = mix64(shared.config.seed ^ mix64(id.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    InFlight {
        id,
        game: Game::from_board(Board::new()),
        rng: StdRng::seed_from_u64(seed),
        phase: GamePhase::NeedRootExpand,
        pending: None,
        leaf_ctx: None,
        staged_enc: Vec::new(),
    }
}

// --- priority queue entry (max-heap by ply, id as total-order tie-break) -----
struct Queued {
    key: u64,
    seq: u64,
    item: Box<InFlight>,
}
impl PartialEq for Queued {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key && self.seq == o.seq
    }
}
impl Eq for Queued {}
impl Ord for Queued {
    fn cmp(&self, o: &Self) -> Ordering {
        self.key.cmp(&o.key).then_with(|| self.seq.cmp(&o.seq))
    }
}
impl PartialOrd for Queued {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

// --- two-bucket inference staging -------------------------------------------
struct Bucket {
    enc: Vec<f32>,             // len = games.len() * ENC_LEN, row-major
    games: Vec<Box<InFlight>>, // row r's game travels WITH the bucket
}
impl Bucket {
    fn with_capacity(b: usize) -> Self {
        Bucket {
            enc: Vec::with_capacity(b * ENC_LEN),
            games: Vec::with_capacity(b),
        }
    }
    fn len(&self) -> usize {
        self.games.len()
    }
}

struct BucketState {
    active: Bucket,        // workers append here
    ready: Option<Bucket>, // a full bucket waiting for the consumer
}

struct Config {
    sims: usize,
    add_root_noise: bool,
    bucket_size: usize,
    seed: u64,
}

struct Shared {
    queue: Mutex<BinaryHeap<Queued>>,
    buckets: Mutex<BucketState>,
    bucket_cv_worker: Condvar,   // workers wait here when both buckets are full
    bucket_cv_consumer: Condvar, // consumer waits here for a full bucket
    stop: AtomicBool,
    id_counter: AtomicU64,
    config: Config,
}

/// Rows with z from each position's side-to-move POV.
fn finalize_rows(g: &mut Game) -> Vec<Row> {
    let z0 = g.result as f32;
    let hist = std::mem::take(&mut g.history);
    hist.into_iter()
        .map(|(state, pi, turn)| (state, pi, if turn == 0 { z0 } else { -z0 }))
        .collect()
}

/// Pure-CPU move step: record (state, pi, turn), sample + push, advance the
/// subtree, mark done on a win or the ply cap.
fn step_move(g: &mut InFlight) {
    let temp = g.game.temperature();
    let pi = visits_to_pi(&g.game.arena, 0, temp);
    let state = g.game.board.encode();
    let turn = g.game.board.turn();
    g.game.history.push((state, pi, turn));

    let handle = sample_move(&g.game.arena, 0, temp, &mut g.rng);
    g.game.board.push(handle);
    advance_subtree(&mut g.game, handle);

    if let Some(res) = g.game.board.outcome() {
        g.game.result = res;
        g.game.done = true;
    }
}

/// Apply the consumer-attached NN result to the tree, then advance the cursor.
fn apply_result(g: &mut InFlight, res: EvalResult, config: &Config) {
    let ctx = g.leaf_ctx.take().expect("pending result without leaf_ctx");
    match ctx.kind {
        LeafKind::RootExpand => {
            expand_node(&mut g.game.arena, 0, &ctx.board, &res.logits);
            if config.add_root_noise {
                add_dirichlet_noise(&mut g.game.arena, 0, &mut g.rng);
            }
            g.phase = GamePhase::Simulating(0);
        }
        LeafKind::SimLeaf => {
            let leaf = *ctx.path.last().unwrap();
            expand_node(&mut g.game.arena, leaf, &ctx.board, &res.logits);
            backprop(&mut g.game.arena, &ctx.path, res.value as f64);
            if let GamePhase::Simulating(s) = g.phase {
                g.phase = GamePhase::Simulating(s + 1);
            }
        }
    }
}

enum Advance {
    Eval,           // g.leaf_ctx is Some, g.staged_enc holds the encoded board
    Done(Vec<Row>), // game completed; rows to submit
}

/// Advance the game until it produces ONE leaf eval or completes. Terminal sims
/// (no eval) are processed inline.
fn advance_until_eval_or_done(g: &mut InFlight, config: &Config) -> Advance {
    loop {
        match g.phase {
            GamePhase::NeedRootExpand => {
                if g.game.arena[0].expanded {
                    // reused subtree root: no eval, just (idempotent) root noise
                    if config.add_root_noise {
                        add_dirichlet_noise(&mut g.game.arena, 0, &mut g.rng);
                    }
                    g.phase = GamePhase::Simulating(0);
                    continue;
                }
                let board = g.game.board.clone();
                g.staged_enc = board.encode();
                g.leaf_ctx = Some(LeafCtx {
                    kind: LeafKind::RootExpand,
                    path: vec![0],
                    board,
                });
                return Advance::Eval;
            }
            GamePhase::Simulating(s) => {
                if s >= config.sims {
                    step_move(g);
                    if g.game.done {
                        return Advance::Done(finalize_rows(&mut g.game));
                    }
                    g.phase = GamePhase::NeedRootExpand;
                    continue;
                }
                let mut board = g.game.board.clone();
                let path = walk_to_leaf(&g.game.arena, 0, &mut board);
                match board.terminal_value() {
                    Some(t) => {
                        backprop(&mut g.game.arena, &path, t as f64);
                        g.phase = GamePhase::Simulating(s + 1);
                        continue;
                    }
                    None => {
                        g.staged_enc = board.encode();
                        g.leaf_ctx = Some(LeafCtx {
                            kind: LeafKind::SimLeaf,
                            path,
                            board,
                        });
                        return Advance::Eval;
                    }
                }
            }
        }
    }
}

/// Push a staged game into the active bucket. Blocks only when BOTH buckets are
/// full (backpressure). On shutdown, drops the game.
fn push_to_bucket(shared: &Shared, g: Box<InFlight>) {
    let b = shared.config.bucket_size;
    let mut bs = shared.buckets.lock().unwrap();
    loop {
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return;
        }
        if bs.active.len() < b {
            bs.active.enc.extend_from_slice(&g.staged_enc);
            bs.active.games.push(g);
            if bs.active.len() == b && bs.ready.is_none() {
                let full = std::mem::replace(&mut bs.active, Bucket::with_capacity(b));
                bs.ready = Some(full);
                shared.bucket_cv_consumer.notify_one();
            }
            return;
        }
        if bs.ready.is_none() {
            let full = std::mem::replace(&mut bs.active, Bucket::with_capacity(b));
            bs.ready = Some(full);
            shared.bucket_cv_consumer.notify_one();
            continue;
        }
        bs = shared.bucket_cv_worker.wait(bs).unwrap();
    }
}

/// Hand a finished game's rows to the Python `submit_game(rows)` callback.
fn submit_completed(submit_game: &PyObject, rows: Vec<Row>) {
    if rows.is_empty() {
        return;
    }
    let _ = Python::with_gil(|py| -> PyResult<()> {
        let pyrows: Vec<_> = rows
            .into_iter()
            .map(|(s, p, z)| {
                let s = Array3::from_shape_vec((INPUT_CHANNELS, SIZE, SIZE), s)
                    .unwrap()
                    .into_pyarray(py);
                let p = Array1::from_vec(p).into_pyarray(py);
                (s, p, z)
            })
            .collect();
        let list = PyList::new(py, pyrows)?;
        submit_game.call1(py, (list,))?;
        Ok(())
    });
}

fn worker_loop(shared: Arc<Shared>, submit_game: PyObject) {
    loop {
        if shared.stop.load(AtomicOrdering::Relaxed) {
            return;
        }
        let mut g: Box<InFlight> = {
            let mut q = shared.queue.lock().unwrap();
            match q.pop() {
                Some(qd) => qd.item,
                None => {
                    drop(q);
                    Box::new(spawn_fresh(&shared))
                }
            }
        };
        if let Some(res) = g.pending.take() {
            apply_result(&mut g, res, &shared.config);
        }
        match advance_until_eval_or_done(&mut g, &shared.config) {
            Advance::Eval => push_to_bucket(&shared, g),
            Advance::Done(rows) => submit_completed(&submit_game, rows),
        }
    }
}

/// Consumer: the only thread running the GPU forward. Polls `stop` with a
/// bounded wait so a never-filling bucket can't wedge shutdown.
fn consumer_loop(shared: Arc<Shared>, forward: PyObject, stop: PyObject) -> PyResult<()> {
    loop {
        let bucket = loop {
            {
                let mut bs = shared.buckets.lock().unwrap();
                if shared.stop.load(AtomicOrdering::Relaxed) {
                    return Ok(());
                }
                if let Some(b) = bs.ready.take() {
                    shared.bucket_cv_worker.notify_all();
                    break b;
                }
                let _ = shared
                    .bucket_cv_consumer
                    .wait_timeout(bs, Duration::from_millis(250))
                    .unwrap();
            }
            if shared.stop.load(AtomicOrdering::Relaxed) {
                return Ok(());
            }
            if poll_stop(&shared, &stop)? {
                return Ok(());
            }
        };

        let Bucket { enc, games } = bucket;
        let m = games.len();
        let (logits_flat, values_flat) =
            Python::with_gil(|py| -> PyResult<(Vec<f32>, Vec<f32>)> {
                let arr = Array4::from_shape_vec((m, INPUT_CHANNELS, SIZE, SIZE), enc)
                    .unwrap()
                    .into_pyarray(py);
                let out = forward.call1(py, (arr,))?;
                let bound = out.bind(py);
                let (logits, values): (PyReadonlyArray2<f32>, PyReadonlyArray1<f32>) =
                    bound.extract()?;
                Ok((logits.as_slice()?.to_vec(), values.as_slice()?.to_vec()))
            })?;

        {
            let mut q = shared.queue.lock().unwrap();
            for (r, mut g) in games.into_iter().enumerate() {
                g.pending = Some(EvalResult {
                    logits: logits_flat[r * POLICY_SIZE..(r + 1) * POLICY_SIZE].to_vec(),
                    value: values_flat[r],
                });
                let key = g.game.history.len() as u64;
                let seq = g.id;
                q.push(Queued { key, seq, item: g });
            }
        }

        if poll_stop(&shared, &stop)? {
            return Ok(());
        }
    }
}

fn poll_stop(shared: &Shared, stop: &PyObject) -> PyResult<bool> {
    let should =
        Python::with_gil(|py| -> PyResult<bool> { stop.call0(py)?.bind(py).extract::<bool>() })?;
    if should {
        shared.stop.store(true, AtomicOrdering::Relaxed);
        shared.bucket_cv_worker.notify_all();
        shared.bucket_cv_consumer.notify_all();
    }
    Ok(should)
}

#[pyclass]
pub struct PipelineEngine {
    sims: usize,
    seed: u64,
    add_root_noise: bool,
    n_threads: usize,
    bucket_size: usize,
}

#[pymethods]
impl PipelineEngine {
    #[new]
    fn new(sims: usize, seed: u64, add_root_noise: bool, n_threads: usize, bucket_size: usize) -> Self {
        PipelineEngine {
            sims,
            seed,
            add_root_noise,
            n_threads: n_threads.max(1),
            bucket_size: bucket_size.max(1),
        }
    }

    /// Drive the self-play pipeline until `stop()` is true.
    ///
    /// `forward(batch[M,8,9,9] f32) -> (logits[M,140] f32, values[M] f32)` is the
    /// in-process GPU forward (consumer only). `submit_game(rows)` receives one
    /// finished game's list of (state[8,9,9], pi[140], z). In-flight games are
    /// dropped on shutdown.
    fn run(&mut self, py: Python<'_>, forward: PyObject, submit_game: PyObject, stop: PyObject) -> PyResult<()> {
        let n_threads = self.n_threads;
        let bucket_size = self.bucket_size;
        let shared = Arc::new(Shared {
            queue: Mutex::new(BinaryHeap::new()),
            buckets: Mutex::new(BucketState {
                active: Bucket::with_capacity(bucket_size),
                ready: None,
            }),
            bucket_cv_worker: Condvar::new(),
            bucket_cv_consumer: Condvar::new(),
            stop: AtomicBool::new(false),
            id_counter: AtomicU64::new(0),
            config: Config {
                sims: self.sims,
                add_root_noise: self.add_root_noise,
                bucket_size,
                seed: self.seed,
            },
        });

        let submit_games: Vec<PyObject> = (0..n_threads).map(|_| submit_game.clone_ref(py)).collect();

        py.allow_threads(move || -> PyResult<()> {
            let mut handles = Vec::with_capacity(n_threads);
            for sg in submit_games {
                let sh = shared.clone();
                handles.push(thread::spawn(move || worker_loop(sh, sg)));
            }
            let res = consumer_loop(shared.clone(), forward, stop);
            shared.stop.store(true, AtomicOrdering::Relaxed);
            shared.bucket_cv_worker.notify_all();
            shared.bucket_cv_consumer.notify_all();
            for h in handles {
                let _ = h.join();
            }
            res
        })
    }
}

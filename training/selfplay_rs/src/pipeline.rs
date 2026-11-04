//! Decoupled non-blocking self-play pipeline, 100% Rust (no GIL, no PyO3).
//! Port of chess-ai-engine's `selfplay_rs/src/pipeline.rs`, built from the
//! game-side FSM of the old pyo3 `quoridor_rs/src/pipeline.rs`.
//!
//! The Python-boundary apparatus is gone: no consumer holding the GIL, no numpy
//! round-trip. ONE inference thread owns the AOTI model and a pool of pinned
//! input slots, and talks to the worker threads with plain Rust data.
//!
//! Topology:
//!   - global ply-priority queue of in-flight games (most plies first);
//!   - N worker threads advance a game's FSM until it yields ONE leaf eval (push
//!     to the active bucket) or completes (hand rows to the sink); empty queue =>
//!     spawn a fresh game (dynamic game count, self-regulates to fill the pipe);
//!   - two buckets: workers fill one while the inference thread takes the other;
//!   - ONE inference thread: copies a full bucket into a free pinned slot,
//!     async-copies it to the GPU, launches the fused AOTI forward, records a
//!     CUDA event and hands the in-flight slot to the scatter thread;
//!   - ONE scatter thread: waits the event, copies the bf16 outputs back,
//!     requeues the games with their results and recycles the slot.
//!
//! With N slots, up to N forwards are queued on the GPU back-to-back, so the
//! readback/requeue of one batch overlaps the next batch's forward.
//!
//! Weight hot-reload: the inference thread mtime-polls the safetensors sidecar
//! the trainer publishes and swaps the raw weights into the live AOTI constant
//! buffer (no recompile).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ffi::CString;
use std::io::{self, BufWriter, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use quoridor_core::board::{Board, ENC_LEN, INPUT_CHANNELS, POLICY_SIZE, SIZE};
use quoridor_core::mcts::{
    add_dirichlet_noise, advance_subtree, backprop, expand_node, sample_move, visits_to_pi,
    walk_to_leaf, Game,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tch::{Device, Kind, Tensor};

use crate::aoti::AotiModel;
use crate::cudagraph::{self, CudaEvent};

pub type Row = (Vec<f32>, Vec<f32>, f32); // (state[648], pi[140], z)

const C: i64 = INPUT_CHANNELS as i64;
const S: i64 = SIZE as i64;

// --- per-game seeding (splitmix64; determinism not required, just spread) ---
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// --- per-game FSM (verbatim from the pyo3 pipeline) --------------------------
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
    logits: Arc<Vec<f32>>, // full batch policy [m * POLICY_SIZE]; sliced by `row`
    row: usize,
    value: f32,
}
/// A game in flight. Boxed so it moves queue<->bucket<->worker by pointer.
struct InFlight {
    id: u64,
    game: Game,
    rng: StdRng,
    phase: GamePhase,
    pending: Option<EvalResult>, // written by the scatter thread, consumed by a worker
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
    ready: Option<Bucket>, // a full bucket waiting for the inference thread
}

pub struct Config {
    pub sims: usize,
    pub add_root_noise: bool,
    pub bucket_size: usize,
    pub seed: u64,
    pub n_threads: usize,
    pub n_slots: usize,       // in-flight forwards / pinned input slots
    pub model_path: String,   // serving_model.pt2 (AOTInductor graph, loaded once)
    pub weights_path: String, // serving_weights.safetensors (raw weights swapped in)
    pub reload_every: Duration,
}

struct Shared {
    queue: Mutex<BinaryHeap<Queued>>,
    buckets: Mutex<BucketState>,
    cv_worker: Condvar, // workers wait here when both buckets are full
    cv_infer: Condvar,  // the inference thread waits here for a full bucket
    stop: AtomicBool,
    id_counter: AtomicU64,
    config: Config,
    games_done: AtomicU64,
    plies_done: AtomicU64,
    forwards: AtomicU64,
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

/// Apply the scatter-attached NN result to the tree, then advance the cursor.
fn apply_result(g: &mut InFlight, res: EvalResult, config: &Config) {
    let ctx = g.leaf_ctx.take().expect("pending result without leaf_ctx");
    let logits = &res.logits[res.row * POLICY_SIZE..(res.row + 1) * POLICY_SIZE];
    match ctx.kind {
        LeafKind::RootExpand => {
            expand_node(&mut g.game.arena, 0, &ctx.board, logits);
            if config.add_root_noise {
                add_dirichlet_noise(&mut g.game.arena, 0, &mut g.rng);
            }
            g.phase = GamePhase::Simulating(0);
        }
        LeafKind::SimLeaf => {
            let leaf = *ctx.path.last().unwrap();
            expand_node(&mut g.game.arena, leaf, &ctx.board, logits);
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
                shared.cv_infer.notify_one();
            }
            return;
        }
        if bs.ready.is_none() {
            let full = std::mem::replace(&mut bs.active, Bucket::with_capacity(b));
            bs.ready = Some(full);
            shared.cv_infer.notify_one();
            continue;
        }
        bs = shared.cv_worker.wait(bs).unwrap();
    }
}

fn requeue_scattered(shared: &Shared, logits: Arc<Vec<f32>>, values: &[f32], games: Vec<Box<InFlight>>) {
    let mut q = shared.queue.lock().unwrap();
    for (r, mut g) in games.into_iter().enumerate() {
        g.pending = Some(EvalResult {
            logits: Arc::clone(&logits),
            row: r,
            value: values[r],
        });
        let key = g.game.history.len() as u64;
        let seq = g.id;
        q.push(Queued { key, seq, item: g });
    }
}

/// Worker: pop (or spawn) a game, apply any pending result, advance until it
/// yields one eval (push to bucket) or completes (sink), forever until stop.
fn worker_loop(shared: Arc<Shared>, sink: Arc<dyn Fn(Vec<Row>) + Send + Sync>) {
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
            Advance::Done(rows) => {
                shared.games_done.fetch_add(1, AtomicOrdering::Relaxed);
                shared.plies_done.fetch_add(rows.len() as u64, AtomicOrdering::Relaxed);
                sink(rows);
            }
        }
    }
}

// --- inference + scatter (overlapped) ---------------------------------------
/// bf16 CUDA tensor -> Vec<f32> on host. The D2H copy is bf16 (half the bytes of
/// an f32 upcast-then-copy); the bf16->f32 widening happens on the CPU.
fn bf16_to_vec_f32(t: &Tensor) -> Vec<f32> {
    let c = t.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
    let n = c.numel();
    let mut v = vec![0f32; n];
    c.copy_data(&mut v, n);
    v
}

/// One in-flight forward handed from the inference thread to the scatter thread.
/// Owns its device tensors (Tensor is Send); the scatter thread reads the outputs
/// after `event` and recycles the pinned-input `slot`.
struct WorkItem {
    slot: usize,
    m: usize,
    _x_dev: Tensor, // kept alive until the forward (event) completes
    out_logits: Tensor,
    out_values: Tensor,
    event: CudaEvent,
    games: Vec<Box<InFlight>>,
}

/// Owns the model + a pool of pinned input buffers. The inference thread fills a
/// pinned slot, async-copies it to the device, launches the fused AOTI forward,
/// and records an event, all non-blocking, so forwards queue back-to-back.
struct Infer {
    model: AotiModel,
    dev: Device,
    pinned_in: Vec<Tensor>, // N pinned host bf16 [B,8,9,9]
    weights_path: String,
    reload_every: Duration,
    last_check: Instant,
    last_mtime: Option<std::time::SystemTime>,
}

impl Infer {
    fn new(config: &Config, n_slots: usize) -> Infer {
        let dev = Device::Cuda(0);
        let b = config.bucket_size as i64;
        // The .pt2 defines the forward graph + initial weights and is loaded ONCE;
        // fresh weights arrive via the safetensors sidecar (maybe_reload).
        let model = AotiModel::load(&config.model_path);
        let pinned_in = (0..n_slots)
            .map(|_| Tensor::zeros([b, C, S, S], (Kind::BFloat16, Device::Cpu)).pin_memory(dev))
            .collect();
        Infer {
            model,
            dev,
            pinned_in,
            // None (file absent at startup) still reloads once the file appears.
            last_mtime: std::fs::metadata(&config.weights_path).and_then(|m| m.modified()).ok(),
            weights_path: config.weights_path.clone(),
            reload_every: config.reload_every,
            last_check: Instant::now(),
        }
    }

    fn maybe_reload(&mut self) {
        if self.last_check.elapsed() < self.reload_every {
            return;
        }
        self.last_check = Instant::now();
        let m = std::fs::metadata(&self.weights_path).and_then(|md| md.modified()).ok();
        if m.is_none() || m == self.last_mtime {
            return;
        }
        // Trainer published fresh weights: read the safetensors sidecar, move the raw
        // primaries to CUDA bf16, and push them into the live AOTI constant buffer
        // (update_constant_buffer -> run_const_fold -> swap). The trainer writes the
        // file atomically (rename), so a changed mtime means a complete file.
        let entries = match Tensor::read_safetensors(&self.weights_path) {
            Ok(v) => v,
            Err(_) => return, // transient; retry on the next tick
        };
        let mut names: Vec<CString> = Vec::with_capacity(entries.len());
        let mut tensors: Vec<Tensor> = Vec::with_capacity(entries.len());
        for (name, t) in entries {
            // AOTI keys constants by the mangled FQN ('.'->'_').
            names.push(CString::new(name.replace('.', "_")).unwrap());
            tensors.push(t.to_device(self.dev).to_kind(Kind::BFloat16));
        }
        let refs: Vec<&Tensor> = tensors.iter().collect();
        self.model.swap_weights(&names, &refs);
        self.last_mtime = m;
    }

    /// Fill pinned slot `slot` with `m` rows, async-H2D, launch the forward, and
    /// record a completion event. Returns owned device tensors + event (all async;
    /// the CPU does not block on the GPU here).
    fn launch(&mut self, slot: usize, enc: &[f32], m: usize) -> (Tensor, Tensor, Tensor, CudaEvent) {
        let src = Tensor::from_slice(enc).reshape([m as i64, C, S, S]); // CPU f32
        self.pinned_in[slot].narrow(0, 0, m as i64).copy_(&src); // f32 -> bf16 into pinned
        // async H2D from pinned memory (does not block the CPU).
        let x = self.pinned_in[slot].internal_to_copy((Kind::BFloat16, self.dev), true);
        let opts = (Kind::BFloat16, self.dev);
        let b = self.pinned_in[slot].size()[0];
        let out_logits = Tensor::zeros([b, POLICY_SIZE as i64], opts);
        let out_values = Tensor::zeros([b], opts);
        self.model.run(&x, &out_logits, &out_values); // async on the current stream
        let event = CudaEvent::new();
        event.record();
        (x, out_logits, out_values, event)
    }
}

/// Producer: pull full buckets, launch forwards back-to-back into free slots, and
/// hand each to the scatter thread. Blocks only when all N slots are in flight.
fn inference_thread(shared: Arc<Shared>, work_tx: SyncSender<WorkItem>, free_rx: Receiver<usize>) {
    cudagraph::set_side_stream(); // a non-default stream for all inference ops
    let mut infer = Infer::new(&shared.config, shared.config.n_slots.max(1));
    loop {
        // acquire a full bucket, polling stop while idle.
        let bucket = loop {
            let mut bs = shared.buckets.lock().unwrap();
            if shared.stop.load(AtomicOrdering::Relaxed) {
                return;
            }
            if let Some(b) = bs.ready.take() {
                shared.cv_worker.notify_all();
                break b;
            }
            let _ = shared.cv_infer.wait_timeout(bs, Duration::from_millis(200)).unwrap();
            if shared.stop.load(AtomicOrdering::Relaxed) {
                return;
            }
        };
        infer.maybe_reload();
        let slot = match free_rx.recv() {
            Ok(s) => s,
            Err(_) => return, // scatter gone
        };
        let Bucket { enc, games } = bucket;
        let m = games.len();
        shared.forwards.fetch_add(1, AtomicOrdering::Relaxed);
        let (x, out_logits, out_values, event) = infer.launch(slot, &enc, m);
        let item = WorkItem { slot, m, _x_dev: x, out_logits, out_values, event, games };
        if work_tx.send(item).is_err() {
            return;
        }
    }
}

/// Consumer: wait each forward's event, read its bf16 outputs (D2H), requeue the
/// games, and recycle the pinned slot. Overlaps the inference thread's next forward.
fn scatter_thread(shared: Arc<Shared>, work_rx: Receiver<WorkItem>, free_tx: SyncSender<usize>) {
    while let Ok(item) = work_rx.recv() {
        item.event.sync(); // wait the forward + output copy
        let m = item.m as i64;
        let logits = Arc::new(bf16_to_vec_f32(&item.out_logits.narrow(0, 0, m)));
        let values = bf16_to_vec_f32(&item.out_values.narrow(0, 0, m));
        requeue_scattered(&shared, logits, &values, item.games);
        if free_tx.send(item.slot).is_err() {
            return;
        }
        // item's device tensors + event drop here (after the readback).
    }
}

fn make_shared(config: Config) -> Arc<Shared> {
    let b = config.bucket_size.max(1);
    Arc::new(Shared {
        queue: Mutex::new(BinaryHeap::new()),
        buckets: Mutex::new(BucketState { active: Bucket::with_capacity(b), ready: None }),
        cv_worker: Condvar::new(),
        cv_infer: Condvar::new(),
        stop: AtomicBool::new(false),
        id_counter: AtomicU64::new(0),
        games_done: AtomicU64::new(0),
        plies_done: AtomicU64::new(0),
        forwards: AtomicU64::new(0),
        config,
    })
}

struct Pipeline {
    workers: Vec<thread::JoinHandle<()>>,
    infer: thread::JoinHandle<()>,
    scatter: thread::JoinHandle<()>,
}

fn spawn_pipeline(shared: Arc<Shared>, sink: Arc<dyn Fn(Vec<Row>) + Send + Sync>) -> Pipeline {
    let n_slots = shared.config.n_slots.max(1);
    let n_threads = shared.config.n_threads.max(1);
    let mut workers = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let sh = shared.clone();
        let sk = sink.clone();
        workers.push(thread::spawn(move || worker_loop(sh, sk)));
    }

    let (work_tx, work_rx) = sync_channel::<WorkItem>(n_slots);
    let (free_tx, free_rx) = sync_channel::<usize>(n_slots);
    for i in 0..n_slots {
        free_tx.send(i).unwrap();
    }
    let infer = {
        let sh = shared.clone();
        thread::spawn(move || inference_thread(sh, work_tx, free_rx))
    };
    let scatter = {
        let sh = shared.clone();
        thread::spawn(move || scatter_thread(sh, work_rx, free_tx))
    };
    Pipeline { workers, infer, scatter }
}

fn shutdown(shared: &Shared, p: Pipeline) {
    shared.stop.store(true, AtomicOrdering::Relaxed);
    {
        let _bs = shared.buckets.lock().unwrap();
        shared.cv_worker.notify_all();
        shared.cv_infer.notify_all();
    }
    // Workers see `stop` at the loop top or in push_to_bucket and exit. The
    // inference thread returns at its stop-check and drops work_tx, ending the
    // scatter thread once its queue drains. Joining workers first is safe:
    // scatter keeps freeing slots, so inference never wedges on a slot.
    for h in p.workers {
        let _ = h.join();
    }
    let _ = p.infer.join();
    let _ = p.scatter.join();
}

/// Bench mode: run for `run` seconds, printing games/sec + avg_plies + rows/sec
/// every `interval` seconds. Exclusive GPU expected.
pub fn run_bench(config: Config, run: Duration, interval: Duration) {
    let shared = make_shared(config);
    let sink: Arc<dyn Fn(Vec<Row>) + Send + Sync> = Arc::new(|_rows: Vec<Row>| {});
    let pipeline = spawn_pipeline(shared.clone(), sink);

    let start = Instant::now();
    let mut win_start = start;
    let (mut last_games, mut last_plies, mut last_fwd) = (0u64, 0u64, 0u64);
    println!("selfplay_rs bench: batch={} threads={} slots={} sims={}",
             shared.config.bucket_size, shared.config.n_threads, shared.config.n_slots, shared.config.sims);
    while start.elapsed() < run {
        thread::sleep(Duration::from_millis(200));
        if win_start.elapsed() >= interval {
            let dt = win_start.elapsed().as_secs_f64();
            let g = shared.games_done.load(AtomicOrdering::Relaxed);
            let p = shared.plies_done.load(AtomicOrdering::Relaxed);
            let f = shared.forwards.load(AtomicOrdering::Relaxed);
            let (dg, dp, df) = (g - last_games, p - last_plies, f - last_fwd);
            println!(
                "[+{:5.0}s] {:6.2} games/sec  ({:5.1} avg plies, {:6.0} rows/sec)  {:5.0} fwd/s",
                start.elapsed().as_secs_f64(),
                dg as f64 / dt,
                dp as f64 / dg.max(1) as f64,
                dp as f64 / dt,
                df as f64 / dt,
            );
            last_games = g;
            last_plies = p;
            last_fwd = f;
            win_start = Instant::now();
        }
    }
    shutdown(&shared, pipeline);
    println!("DONE (read steady-state from the last few intervals)");
}

// --- serve mode (production IPC) ---------------------------------------------
/// Write `&[f32]` as little-endian bytes (aarch64/x86 are LE; matches numpy
/// frombuffer('<f4') on the reader side). Contiguous slice -> raw bytes.
fn write_f32s<W: Write>(w: &mut W, s: &[f32]) -> io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) };
    w.write_all(bytes)
}

/// One finished game = one frame: u32 n_rows, then n_rows x (state[648] f32,
/// pi[140] f32, z f32), all little-endian. Flushed per game so the trainer's
/// reader sees data promptly.
fn write_frame(out: &Mutex<BufWriter<io::Stdout>>, rows: &[Row]) {
    let mut w = out.lock().unwrap();
    let _ = w.write_all(&(rows.len() as u32).to_le_bytes());
    for (state, pi, z) in rows {
        let _ = write_f32s(&mut *w, state);
        let _ = write_f32s(&mut *w, pi);
        let _ = w.write_all(&z.to_le_bytes());
    }
    let _ = w.flush();
}

/// Serve mode: run the pipeline forever, streaming finished games as framed bytes
/// on stdout for the orchestrator's reader thread. Stops when the parent closes
/// our stdin (EOF = orchestrator died) or on SIGTERM (default-kills the process).
pub fn run_serve(config: Config) {
    let shared = make_shared(config);
    let stdout = Arc::new(Mutex::new(BufWriter::new(io::stdout())));
    let so = stdout.clone();
    let sink: Arc<dyn Fn(Vec<Row>) + Send + Sync> = Arc::new(move |rows: Vec<Row>| write_frame(&so, &rows));
    let pipeline = spawn_pipeline(shared.clone(), sink);

    // Block until the orchestrator closes our stdin (its death closes the pipe).
    let mut buf = [0u8; 256];
    loop {
        match io::stdin().read(&mut buf) {
            Ok(0) | Err(_) => break, // EOF / error -> shut down
            Ok(_) => {}              // ignore any bytes
        }
    }
    shutdown(&shared, pipeline);
    let _ = stdout.lock().unwrap().flush();
}

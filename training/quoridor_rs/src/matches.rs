//! Lockstep batched match play for evaluation (training/evaluate.py).
//!
//! Plays `n_games` between player A and player B (alternating who moves first)
//! with a fresh MCTS tree per move and no root noise, mirroring chess-ai-engine's
//! evaluate.py `NetPlayer`/`RandomPlayer`. The first `opening_plies` moves are
//! sampled in proportion to visits for game diversity; after that the most
//! visited move is played. Either player may be the uniform-random player, which
//! never needs the net.
//!
//! Python drives the loop: `leaves()` advances every game to its next NN leaf
//! and returns the stacked encodings plus which player (0 = A, 1 = B) each row
//! belongs to; Python runs each player's net on its rows and hands the results
//! back through `apply()`.

use numpy::ndarray::{Array1, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray4, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use quoridor_core::board::{Board, ENC_LEN, INPUT_CHANNELS, POLICY_SIZE, SIZE};
use quoridor_core::mcts::{backprop, expand_node, sample_move, walk_to_leaf, Node};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

#[pyclass]
pub struct MatchEngine {
    games: Vec<MatchGame>,
    sims: usize,
    opening_plies: usize,
    random: [bool; 2],
    rng: StdRng,
}

impl MatchEngine {
    /// Advance one game until it needs an NN eval (staged in `pending`) or ends.
    fn advance(&mut self, gi: usize) {
        let (sims, opening_plies, random) = (self.sims, self.opening_plies, self.random);
        let g = &mut self.games[gi];
        while g.result.is_none() && g.pending.is_none() {
            if random[g.mover()] {
                let moves = g.board.legal_moves();
                let idx = moves[self.rng.gen_range(0..moves.len())];
                g.play(idx);
                continue;
            }
            if !g.arena[0].expanded {
                g.pending = Some((vec![0], g.board.clone()));
                break;
            }
            if g.sim >= sims {
                let temp = if g.board.ply() < opening_plies { 1.0 } else { 0.0 };
                let idx = sample_move(&g.arena, 0, temp, &mut self.rng);
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
}

#[pymethods]
impl MatchEngine {
    #[new]
    fn new(n_games: usize, sims: usize, opening_plies: usize, seed: u64, a_random: bool, b_random: bool) -> Self {
        let games = (0..n_games)
            .map(|i| MatchGame {
                board: Board::new(),
                a_first: i % 2 == 0,
                arena: vec![Node::new(0.0, -1)],
                sim: 0,
                pending: None,
                result: None,
            })
            .collect();
        MatchEngine {
            games,
            sims: sims.max(1),
            opening_plies,
            random: [a_random, b_random],
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Advance every unfinished game to its next leaf. Returns (encodings
    /// [M,8,9,9], player [M]) for the games awaiting an eval, in game order.
    /// M == 0 means every game is over.
    fn leaves<'py>(&mut self, py: Python<'py>) -> (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray1<i64>>) {
        let mut enc = Vec::new();
        let mut who = Vec::new();
        for gi in 0..self.games.len() {
            self.advance(gi);
            let g = &self.games[gi];
            if let Some((_, b)) = &g.pending {
                enc.extend(b.encode());
                who.push(g.mover() as i64);
            }
        }
        let m = who.len();
        debug_assert_eq!(enc.len(), m * ENC_LEN);
        let enc = Array4::from_shape_vec((m, INPUT_CHANNELS, SIZE, SIZE), enc).unwrap();
        (enc.into_pyarray(py), Array1::from_vec(who).into_pyarray(py))
    }

    /// Apply results for the rows returned by the last `leaves()` call.
    fn apply(&mut self, logits: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>) -> PyResult<()> {
        let logits = logits.as_slice()?;
        let values = values.as_slice()?;
        let mut r = 0;
        for g in self.games.iter_mut() {
            let Some((path, b)) = g.pending.take() else { continue };
            let row = &logits[r * POLICY_SIZE..(r + 1) * POLICY_SIZE];
            let leaf = *path.last().unwrap();
            expand_node(&mut g.arena, leaf, &b, row);
            if path.len() > 1 {
                backprop(&mut g.arena, &path, values[r] as f64);
                g.sim += 1;
            }
            r += 1;
        }
        assert_eq!(r, values.len(), "apply(): row count does not match the last leaves()");
        Ok(())
    }

    /// Per-game results from A's POV (+1 win, 0 draw, −1 loss); None while unfinished.
    fn results(&self) -> Vec<Option<i8>> {
        self.games
            .iter()
            .map(|g| g.result.map(|r| if g.a_first { r } else { -r }))
            .collect()
    }
}

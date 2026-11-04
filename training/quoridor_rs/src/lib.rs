//! Python bindings for the Quoridor core: a `Board` for tests and tooling, and
//! the evaluation `MatchEngine`. Self-play runs in the standalone selfplay_rs
//! binary.

mod matches;

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use quoridor_core::board::{self, Move, INPUT_CHANNELS, SIZE};

#[pyclass(name = "Board")]
#[derive(Clone)]
pub struct PyBoard(board::Board);

#[pymethods]
impl PyBoard {
    #[new]
    fn new() -> Self {
        PyBoard(board::Board::new())
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    /// Policy indices of the legal moves for the side to move.
    fn legal_moves(&self) -> Vec<u16> {
        self.0.legal_moves()
    }

    fn push(&mut self, idx: u16) -> PyResult<()> {
        if !self.0.legal_moves().contains(&idx) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("illegal move index {idx}")));
        }
        self.0.push(idx);
        Ok(())
    }

    /// The absolute move for a policy index: ("pawn", r, c) or ("h"|"v", r, c).
    fn decode(&self, idx: u16) -> (&'static str, u8, u8) {
        match self.0.decode(idx) {
            Move::Pawn { r, c } => ("pawn", r, c),
            Move::Wall { horiz, r, c } => (if horiz { "h" } else { "v" }, r, c),
        }
    }

    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        Array3::from_shape_vec((INPUT_CHANNELS, SIZE, SIZE), self.0.encode())
            .unwrap()
            .into_pyarray(py)
    }

    fn mirrored(&self) -> Self {
        PyBoard(self.0.mirrored())
    }

    /// −1 for the side to move if the previous mover just won, 0 at the ply cap.
    fn terminal_value(&self) -> Option<f32> {
        self.0.terminal_value()
    }

    /// +1 / −1 / 0 from player 0's POV once the game is over.
    fn outcome(&self) -> Option<i8> {
        self.0.outcome()
    }

    #[getter]
    fn turn(&self) -> usize {
        self.0.turn()
    }

    #[getter]
    fn ply(&self) -> usize {
        self.0.ply()
    }

    fn pawn(&self, player: usize) -> (usize, usize) {
        self.0.pawn(player)
    }

    fn walls_left(&self, player: usize) -> usize {
        self.0.walls_left(player)
    }
}

#[pyfunction]
fn mirror_state_perm() -> Vec<usize> {
    board::mirror_state_perm()
}

#[pyfunction]
fn mirror_policy_perm() -> Vec<usize> {
    board::mirror_policy_perm()
}

#[pymodule]
fn quoridor_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<matches::MatchEngine>()?;
    m.add_function(wrap_pyfunction!(mirror_state_perm, m)?)?;
    m.add_function(wrap_pyfunction!(mirror_policy_perm, m)?)?;
    m.add("INPUT_CHANNELS", INPUT_CHANNELS)?;
    m.add("BOARD_SIZE", SIZE)?;
    m.add("POLICY_SIZE", board::POLICY_SIZE)?;
    m.add("ENC_LEN", board::ENC_LEN)?;
    m.add("MAX_PLIES", board::MAX_PLIES)?;
    Ok(())
}

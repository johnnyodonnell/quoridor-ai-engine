//! Pure-Rust Quoridor core: the rules + AlphaZero encoding (`board`) and the
//! MCTS search primitives (`mcts`). No pyo3, so `cargo test` links normally; the
//! `quoridor_rs` extension wraps it for Python.

pub mod board;
pub mod mcts;

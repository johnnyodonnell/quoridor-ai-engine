//! selfplay_rs library surface.
//!
//! `net` is the eager tch QuoridorNet (parity-exact explicit-BN forward plus a
//! BN-folded diagnostic path). The AOTI / CUDA-graph / pipeline machinery used
//! only by the self-play worker stays bin-local to `main.rs`.

pub mod net;

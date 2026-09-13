//! Backend-agnostic ML inference support for `wasmer-wasix`'s `wasi_ephemeral_nn`
//! host functions (the pre-Component-Model core-wasm WASI-NN ABI). This crate
//! defines the [`NnBackend`]/[`NnGraph`]/[`NnExecutionContext`] traits and the
//! wire types they operate on; it has no dependency on `wasmer`/`wasmer-wasix`,
//! so it can be reused (or swapped) independently of the runtime that hosts it.
//!
//! [`CpuStub`] is the default no-op backend. A `candle`-backed reference
//! implementation, gated by this crate's `candle` feature, runs real inference
//! (CPU or CUDA) via the `candle` crate.

mod backend;
mod stub;
mod types;

// `candle_backend` is added once the reference backend lands; see the crate's
// `candle` Cargo feature.

pub use backend::{NnBackend, NnExecutionContext, NnGraph};
pub use stub::CpuStub;
pub use types::{ExecutionTarget, GraphEncoding, NnErrno, Tensor, TensorType};

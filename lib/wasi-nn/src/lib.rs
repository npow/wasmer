//! Backend-agnostic ML inference support for `wasmer-wasix`'s `wasi_ephemeral_nn`
//! host functions (the pre-Component-Model core-wasm WASI-NN ABI). This crate
//! defines the [`NnBackend`]/[`NnGraph`]/[`NnExecutionContext`] traits and the
//! wire types they operate on; it has no dependency on `wasmer`/`wasmer-wasix`,
//! so it can be reused (or swapped) independently of the runtime that hosts it.
//!
//! [`CpuStub`] is the default no-op backend. [`candle_backend::CandleBackend`],
//! gated by this crate's `candle` feature (on by default), runs real inference
//! (CPU or CUDA) via the `candle` crate.

mod backend;
mod stub;
mod types;

#[cfg(feature = "candle")]
pub mod candle_backend;

pub use backend::{NnBackend, NnExecutionContext, NnGraph};
pub use stub::CpuStub;
pub use types::{ExecutionTarget, GraphEncoding, NnErrno, Tensor, TensorType};

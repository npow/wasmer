//! Backend-agnostic traits implemented by an inference engine (e.g. the `candle`
//! reference backend in [`crate::candle_backend`]). `wasmer-wasix` depends only on
//! this trait, never on a concrete engine, so swapping backends touches no syscall
//! code.

use crate::types::{ExecutionTarget, GraphEncoding, NnErrno, Tensor};

/// A loaded model, capable of spawning execution contexts.
pub trait NnGraph: Send + Sync + std::fmt::Debug {
    fn init_execution_context(&self) -> Result<Box<dyn NnExecutionContext>, NnErrno>;
}

/// One in-flight (or reusable) inference run over a [`NnGraph`].
pub trait NnExecutionContext: Send + Sync + std::fmt::Debug {
    fn set_input(&mut self, index: u32, tensor: Tensor) -> Result<(), NnErrno>;
    fn compute(&mut self) -> Result<(), NnErrno>;
    /// Returns the raw little-endian output bytes for the given output index.
    fn get_output(&self, index: u32) -> Result<Vec<u8>, NnErrno>;
}

/// An inference engine. One process may only ever have one backend configured
/// (set once via the embedder/CLI), matching wasi-nn's single-implementation-
/// per-host model.
pub trait NnBackend: Send + Sync + std::fmt::Debug {
    /// Load a graph from one or more raw model byte blobs (`$graph_builder_array`).
    fn load(
        &self,
        builders: &[Vec<u8>],
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Box<dyn NnGraph>, NnErrno>;

    /// Load a graph the embedder pre-registered under a name. Unlike `load`, this
    /// is not spec-mandated to be implemented; the default rejects with `NotFound`.
    fn load_by_name(&self, _name: &str) -> Result<Box<dyn NnGraph>, NnErrno> {
        Err(NnErrno::NotFound)
    }
}

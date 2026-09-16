//! Default backend: rejects every graph load with `UnsupportedOperation`. Used
//! when no real backend (e.g. `candle`) has been wired up by the embedder, so
//! `wasi_ephemeral_nn` host functions always have *some* backend to call rather
//! than needing a separate "no backend configured" case at every call site.

use crate::backend::{NnBackend, NnExecutionContext, NnGraph};
use crate::types::{ExecutionTarget, GraphEncoding, NnErrno};

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuStub;

impl NnBackend for CpuStub {
    fn load(
        &self,
        _builders: &[Vec<u8>],
        _encoding: GraphEncoding,
        _target: ExecutionTarget,
    ) -> Result<Box<dyn NnGraph>, NnErrno> {
        Err(NnErrno::UnsupportedOperation)
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct UnreachableGraph;

impl NnGraph for UnreachableGraph {
    fn init_execution_context(&self) -> Result<Box<dyn NnExecutionContext>, NnErrno> {
        Err(NnErrno::UnsupportedOperation)
    }
}

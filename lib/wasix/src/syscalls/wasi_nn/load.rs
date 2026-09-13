use super::types::WasmGraphBuilder;
use crate::syscalls::*;

/// ### `load()`
/// Loads a graph (i.e. model) from one or more raw byte blobs, returning an
/// opaque `graph` handle through `graph_out`.
///
/// Gated by `Capabilities::nn`: denied outright unless `allow`; `target == gpu`
/// additionally requires `allow_gpu`.
///
/// TODO(wasi-nn): body -- see `docs/CONTRIBUTING.md` sibling-first rule; this is
/// intentionally left returning `unsupported_operation` until capability
/// enforcement + backend dispatch land.
#[instrument(level = "trace", skip_all)]
pub fn nn_load(
    mut _ctx: FunctionEnvMut<'_, WasiEnv>,
    _builder_ptr: WasmPtr<WasmGraphBuilder>,
    _builder_len: u32,
    _encoding: u32,
    _target: u32,
    _graph_out: WasmPtr<u32>,
) -> u32 {
    wasmer_wasi_nn::NnErrno::UnsupportedOperation.to_u32()
}

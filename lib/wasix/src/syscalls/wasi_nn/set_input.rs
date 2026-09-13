use super::types::WasmTensor;
use crate::syscalls::*;

/// ### `set_input()`
/// Copies a `tensor` (dimensions + type + raw element bytes, all read from
/// guest memory) into execution context `context` at input `index`.
///
/// TODO(wasi-nn): body.
#[instrument(level = "trace", skip_all)]
pub fn nn_set_input(
    mut _ctx: FunctionEnvMut<'_, WasiEnv>,
    _context: u32,
    _index: u32,
    _tensor_ptr: WasmPtr<WasmTensor>,
) -> u32 {
    wasmer_wasi_nn::NnErrno::UnsupportedOperation.to_u32()
}

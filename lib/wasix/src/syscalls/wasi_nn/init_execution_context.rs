use crate::syscalls::*;

/// ### `init_execution_context()`
/// Spawns a `graph_execution_context` handle from an already-loaded `graph`.
///
/// TODO(wasi-nn): body.
#[instrument(level = "trace", skip_all)]
pub fn nn_init_execution_context(
    mut _ctx: FunctionEnvMut<'_, WasiEnv>,
    _graph: u32,
    _context_out: WasmPtr<u32>,
) -> u32 {
    wasmer_wasi_nn::NnErrno::UnsupportedOperation.to_u32()
}

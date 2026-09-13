use crate::syscalls::*;

/// ### `load_by_name()`
/// Loads a graph the embedder pre-registered under `name`. Not spec-mandated;
/// [`wasmer_wasi_nn::NnBackend::load_by_name`] defaults to `not_found`.
///
/// TODO(wasi-nn): body.
#[instrument(level = "trace", skip_all)]
pub fn nn_load_by_name(
    mut _ctx: FunctionEnvMut<'_, WasiEnv>,
    _name_ptr: WasmPtr<u8>,
    _name_len: u32,
    _graph_out: WasmPtr<u32>,
) -> u32 {
    wasmer_wasi_nn::NnErrno::UnsupportedOperation.to_u32()
}

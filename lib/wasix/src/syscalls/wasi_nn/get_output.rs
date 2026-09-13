use crate::syscalls::*;

/// ### `get_output()`
/// Copies the raw output bytes at `index` into guest memory at `out_buffer`
/// (capacity `out_buffer_max_size`), writing the number of bytes written
/// through `bytes_written_out`. Returns `too_large` if the output doesn't fit.
///
/// TODO(wasi-nn): body.
#[instrument(level = "trace", skip_all)]
pub fn nn_get_output(
    mut _ctx: FunctionEnvMut<'_, WasiEnv>,
    _context: u32,
    _index: u32,
    _out_buffer: WasmPtr<u8>,
    _out_buffer_max_size: u32,
    _bytes_written_out: WasmPtr<u32>,
) -> u32 {
    wasmer_wasi_nn::NnErrno::UnsupportedOperation.to_u32()
}

use crate::syscalls::*;
use wasmer_wasi_nn::NnErrno;

/// ### `get_output()`
/// Copies the raw output bytes at `index` into guest memory at `out_buffer`
/// (capacity `out_buffer_max_size`), writing the number of bytes written
/// through `bytes_written_out`. Returns `too_large` if the output doesn't fit.
#[instrument(level = "trace", skip_all)]
pub fn nn_get_output(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    context: u32,
    index: u32,
    out_buffer: WasmPtr<u8>,
    out_buffer_max_size: u32,
    bytes_written_out: WasmPtr<u32>,
) -> u32 {
    match body(
        &ctx,
        context,
        index,
        out_buffer,
        out_buffer_max_size,
        bytes_written_out,
    ) {
        Ok(()) => NnErrno::Success.to_u32(),
        Err(err) => err.to_u32(),
    }
}

fn body(
    ctx: &FunctionEnvMut<'_, WasiEnv>,
    context: u32,
    index: u32,
    out_buffer: WasmPtr<u8>,
    out_buffer_max_size: u32,
    bytes_written_out: WasmPtr<u32>,
) -> Result<(), NnErrno> {
    let env = ctx.data();

    // Drop the lock before touching guest memory -- the output is copied out
    // as an owned `Vec` first, so nothing below needs the execution context.
    let output = {
        let nn = env.state.nn.lock().unwrap();
        let exec_ctx = nn.context(context).ok_or(NnErrno::NotFound)?;
        exec_ctx.get_output(index)?
    };

    let written = u32::try_from(output.len()).map_err(|_| NnErrno::TooLarge)?;
    if written > out_buffer_max_size {
        return Err(NnErrno::TooLarge);
    }

    let memory = unsafe { env.memory_view(ctx) };
    out_buffer
        .slice(&memory, written)
        .and_then(|slice| slice.write_slice(&output))
        .map_err(|_| NnErrno::MissingMemory)?;
    bytes_written_out
        .write(&memory, written)
        .map_err(|_| NnErrno::MissingMemory)?;

    Ok(())
}

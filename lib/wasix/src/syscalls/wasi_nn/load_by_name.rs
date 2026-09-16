use wasmer_wasi_nn::NnErrno;

use crate::syscalls::*;

/// ### `load_by_name()`
/// Loads a graph the embedder pre-registered under `name`. Not spec-mandated;
/// [`wasmer_wasi_nn::NnBackend::load_by_name`] defaults to `not_found`.
///
/// Gated by `Capabilities::nn`: denied outright unless `allow`.
#[instrument(level = "trace", skip_all)]
pub fn nn_load_by_name(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    name_ptr: WasmPtr<u8>,
    name_len: u32,
    graph_out: WasmPtr<u32>,
) -> u32 {
    match load_by_name(ctx, name_ptr, name_len, graph_out) {
        Ok(()) => NnErrno::Success.to_u32(),
        Err(e) => e.to_u32(),
    }
}

fn load_by_name(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    name_ptr: WasmPtr<u8>,
    name_len: u32,
    graph_out: WasmPtr<u32>,
) -> Result<(), NnErrno> {
    if !ctx.data().capabilities.nn.allow {
        return Err(NnErrno::UnsupportedOperation);
    }

    let env = ctx.data();
    let memory = unsafe { env.memory_view(&ctx) };
    let name_bytes = name_ptr
        .slice(&memory, name_len)
        .map_err(|_| NnErrno::MissingMemory)?
        .read_to_vec()
        .map_err(|_| NnErrno::MissingMemory)?;
    let name = String::from_utf8(name_bytes).map_err(|_| NnErrno::InvalidArgument)?;

    let graph = env.state.nn_backend.load_by_name(&name)?;
    let handle = env.state.nn.lock().unwrap().insert_graph(graph)?;

    graph_out
        .write(&memory, handle)
        .map_err(|_| NnErrno::MissingMemory)?;

    Ok(())
}

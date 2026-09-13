use wasmer_wasi_nn::NnErrno;

use crate::syscalls::*;

/// ### `init_execution_context()`
/// Spawns a `graph_execution_context` handle from an already-loaded `graph`.
///
/// No capability check here: the graph already passed the `Capabilities::nn`
/// gate when it was loaded.
#[instrument(level = "trace", skip_all)]
pub fn nn_init_execution_context(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    graph: u32,
    context_out: WasmPtr<u32>,
) -> u32 {
    match init_execution_context(ctx, graph, context_out) {
        Ok(()) => NnErrno::Success.to_u32(),
        Err(e) => e.to_u32(),
    }
}

fn init_execution_context(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    graph: u32,
    context_out: WasmPtr<u32>,
) -> Result<(), NnErrno> {
    let env = ctx.data();

    let exec_ctx = {
        let nn = env.state.nn.lock().unwrap();
        let graph_ref = nn.graph(graph).ok_or(NnErrno::NotFound)?;
        graph_ref.init_execution_context()?
    };
    let handle = env.state.nn.lock().unwrap().insert_context(exec_ctx)?;

    let memory = unsafe { env.memory_view(&ctx) };
    context_out
        .write(&memory, handle)
        .map_err(|_| NnErrno::MissingMemory)?;

    Ok(())
}

use wasmer_wasi_nn::{ExecutionTarget, GraphEncoding, NnErrno};

use super::types::WasmGraphBuilder;
use crate::syscalls::*;

/// ### `load()`
/// Loads a graph (i.e. model) from one or more raw byte blobs, returning an
/// opaque `graph` handle through `graph_out`.
///
/// Gated by `Capabilities::nn`: denied outright unless `allow`; `target == gpu`
/// additionally requires `allow_gpu`.
#[instrument(level = "trace", skip_all)]
pub fn nn_load(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    builder_ptr: WasmPtr<WasmGraphBuilder>,
    builder_len: u32,
    encoding: u32,
    target: u32,
    graph_out: WasmPtr<u32>,
) -> u32 {
    match load(ctx, builder_ptr, builder_len, encoding, target, graph_out) {
        Ok(()) => NnErrno::Success.to_u32(),
        Err(e) => e.to_u32(),
    }
}

/// Reads the `$graph_builder_array` starting at `builder_ptr`, i.e. `builder_len`
/// many `(ptr, len)` pairs, and returns the raw bytes each one points to.
fn read_blobs(
    memory: &MemoryView,
    builder_ptr: WasmPtr<WasmGraphBuilder>,
    builder_len: u32,
) -> Result<Vec<Vec<u8>>, NnErrno> {
    let builders = builder_ptr
        .slice(memory, builder_len)
        .map_err(|_| NnErrno::MissingMemory)?
        .read_to_vec()
        .map_err(|_| NnErrno::MissingMemory)?;

    builders
        .into_iter()
        .map(|b| {
            WasmPtr::<u8>::new(b.ptr)
                .slice(memory, b.len)
                .map_err(|_| NnErrno::MissingMemory)?
                .read_to_vec()
                .map_err(|_| NnErrno::MissingMemory)
        })
        .collect()
}

fn load(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    builder_ptr: WasmPtr<WasmGraphBuilder>,
    builder_len: u32,
    encoding: u32,
    target: u32,
    graph_out: WasmPtr<u32>,
) -> Result<(), NnErrno> {
    if !ctx.data().capabilities.nn.allow {
        return Err(NnErrno::UnsupportedOperation);
    }

    let encoding =
        GraphEncoding::try_from(u8::try_from(encoding).map_err(|_| NnErrno::InvalidArgument)?)?;
    let target =
        ExecutionTarget::try_from(u8::try_from(target).map_err(|_| NnErrno::InvalidArgument)?)?;
    if target == ExecutionTarget::Gpu && !ctx.data().capabilities.nn.allow_gpu {
        return Err(NnErrno::UnsupportedOperation);
    }

    let env = ctx.data();
    let memory = unsafe { env.memory_view(&ctx) };
    let blobs = read_blobs(&memory, builder_ptr, builder_len)?;

    if let Some(max) = env.capabilities.nn.max_model_bytes {
        let total: u64 = blobs.iter().map(|b| b.len() as u64).sum();
        if total > max {
            return Err(NnErrno::TooLarge);
        }
    }

    let graph = env.state.nn_backend.load(&blobs, encoding, target)?;
    let handle = env.state.nn.lock().unwrap().insert_graph(graph)?;

    graph_out
        .write(&memory, handle)
        .map_err(|_| NnErrno::MissingMemory)?;

    Ok(())
}

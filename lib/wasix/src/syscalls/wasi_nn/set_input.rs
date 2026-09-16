use super::types::WasmTensor;
use crate::syscalls::*;
use wasmer_wasi_nn::{NnErrno, Tensor, TensorType};

/// ### `set_input()`
/// Copies a `tensor` (dimensions + type + raw element bytes, all read from
/// guest memory) into execution context `context` at input `index`.
#[instrument(level = "trace", skip_all)]
pub fn nn_set_input(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    context: u32,
    index: u32,
    tensor_ptr: WasmPtr<WasmTensor>,
) -> u32 {
    match body(&ctx, context, index, tensor_ptr) {
        Ok(()) => NnErrno::Success.to_u32(),
        Err(err) => err.to_u32(),
    }
}

fn body(
    ctx: &FunctionEnvMut<'_, WasiEnv>,
    context: u32,
    index: u32,
    tensor_ptr: WasmPtr<WasmTensor>,
) -> Result<(), NnErrno> {
    let env = ctx.data();
    let memory = unsafe { env.memory_view(ctx) };

    // `MemoryAccessError` and `NnErrno` are both foreign to this crate, so a
    // `From` impl between them would violate the orphan rule -- map inline.
    let wasm_tensor = tensor_ptr
        .read(&memory)
        .map_err(|_| NnErrno::MissingMemory)?;
    let ty = TensorType::try_from(wasm_tensor.ty)?;
    let dimensions = WasmPtr::<u32>::new(wasm_tensor.dimensions_ptr)
        .slice(&memory, wasm_tensor.dimensions_len)
        .and_then(WasmSlice::read_to_vec)
        .map_err(|_| NnErrno::MissingMemory)?;
    let data = WasmPtr::<u8>::new(wasm_tensor.data_ptr)
        .slice(&memory, wasm_tensor.data_len)
        .and_then(WasmSlice::read_to_vec)
        .map_err(|_| NnErrno::MissingMemory)?;
    let tensor = Tensor {
        dimensions,
        ty,
        data,
    };

    let mut nn = env.state.nn.lock().unwrap();
    let exec_ctx = nn.context_mut(context).ok_or(NnErrno::NotFound)?;
    exec_ctx.set_input(index, tensor)
}

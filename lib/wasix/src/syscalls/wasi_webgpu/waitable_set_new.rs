use crate::state::WebgpuErrno;
use crate::syscalls::*;

/// ### `waitable_set_new()`
/// Allocates a new, empty waitable set and returns its handle through
/// `out_set_ptr`.
#[instrument(level = "trace", skip_all)]
pub fn waitable_set_new(ctx: FunctionEnvMut<'_, WasiEnv>, out_set_ptr: i32) -> i32 {
    match waitable_set_new_inner(ctx, out_set_ptr) {
        Ok(()) => WebgpuErrno::Success.to_i32(),
        Err(e) => e.to_i32(),
    }
}

fn waitable_set_new_inner(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    out_set_ptr: i32,
) -> Result<(), WebgpuErrno> {
    if !ctx.data().capabilities.webgpu_spike.allow {
        return Err(WebgpuErrno::CapabilityDenied);
    }

    let env = ctx.data();
    let handle = env.state.webgpu.lock().unwrap().new_waitable_set()?;

    let memory = unsafe { env.memory_view(&ctx) };
    let out: WasmPtr<u32> = WasmPtr::new(out_set_ptr as u32);
    out.write(&memory, handle)
        .map_err(|_| WebgpuErrno::MissingMemory)
}

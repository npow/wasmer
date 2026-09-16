use crate::state::WebgpuErrno;
use crate::syscalls::*;

/// ### `future_read()`
/// Non-blocking read of `future`'s value. If it has already resolved, writes
/// the value to `out_value_ptr` and returns [`WebgpuErrno::Success`]. If
/// still pending, returns [`WebgpuErrno::Blocked`] and writes nothing --
/// the guest must then `waitable_join` `future` into a waitable set (a
/// future is itself a waitable, exactly like a subtask) and
/// `waitable_set_wait` on it. Returns [`WebgpuErrno::BadHandle`] if `future`
/// doesn't exist or names a subtask instead.
#[instrument(level = "trace", skip_all)]
pub fn future_read(ctx: FunctionEnvMut<'_, WasiEnv>, future: i32, out_value_ptr: i32) -> i32 {
    match future_read_inner(ctx, future, out_value_ptr) {
        Ok(()) => WebgpuErrno::Success.to_i32(),
        Err(e) => e.to_i32(),
    }
}

fn future_read_inner(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    future: i32,
    out_value_ptr: i32,
) -> Result<(), WebgpuErrno> {
    if !ctx.data().capabilities.webgpu_spike.allow {
        return Err(WebgpuErrno::CapabilityDenied);
    }

    let future = u32::try_from(future).map_err(|_| WebgpuErrno::BadHandle)?;
    let env = ctx.data();
    let value = env
        .state
        .webgpu
        .lock()
        .unwrap()
        .read_future(future)?
        .ok_or(WebgpuErrno::Blocked)?;

    let memory = unsafe { env.memory_view(&ctx) };
    let out: WasmPtr<u32> = WasmPtr::new(out_value_ptr as u32);
    out.write(&memory, value)
        .map_err(|_| WebgpuErrno::MissingMemory)
}

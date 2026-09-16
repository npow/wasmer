use crate::state::{WaitableKind, WebgpuErrno};
use crate::syscalls::*;

/// ### `future_new()`
/// Allocates a new future in the empty/pending state and returns its handle
/// through `out_future_ptr`. No producer is attached yet -- see
/// `future_resolve_after` for how this bridge fills a future in, since it
/// has no separate host/guest writer actor to call a guest-facing
/// `future_write` on the guest's behalf.
#[instrument(level = "trace", skip_all)]
pub fn future_new(ctx: FunctionEnvMut<'_, WasiEnv>, out_future_ptr: i32) -> i32 {
    match future_new_inner(ctx, out_future_ptr) {
        Ok(()) => WebgpuErrno::Success.to_i32(),
        Err(e) => e.to_i32(),
    }
}

fn future_new_inner(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    out_future_ptr: i32,
) -> Result<(), WebgpuErrno> {
    if !ctx.data().capabilities.webgpu_spike.allow {
        return Err(WebgpuErrno::CapabilityDenied);
    }

    let env = ctx.data();
    let handle = env
        .state
        .webgpu
        .lock()
        .unwrap()
        .insert_pending(WaitableKind::Future)?;

    let memory = unsafe { env.memory_view(&ctx) };
    let out: WasmPtr<u32> = WasmPtr::new(out_future_ptr as u32);
    out.write(&memory, handle)
        .map_err(|_| WebgpuErrno::MissingMemory)
}

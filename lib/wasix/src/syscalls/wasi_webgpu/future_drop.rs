use crate::state::{WaitableKind, WebgpuErrno};
use crate::syscalls::*;

/// ### `future_drop()`
/// Releases future handle `future`. Errors with [`WebgpuErrno::BadHandle`]
/// if `future` names a subtask instead (see `subtask_drop`). Safe to call
/// whether or not the future has been resolved yet.
#[instrument(level = "trace", skip_all)]
pub fn future_drop(ctx: FunctionEnvMut<'_, WasiEnv>, future: i32) -> i32 {
    match future_drop_inner(ctx, future) {
        Ok(()) => WebgpuErrno::Success.to_i32(),
        Err(e) => e.to_i32(),
    }
}

fn future_drop_inner(ctx: FunctionEnvMut<'_, WasiEnv>, future: i32) -> Result<(), WebgpuErrno> {
    if !ctx.data().capabilities.webgpu_spike.allow {
        return Err(WebgpuErrno::CapabilityDenied);
    }

    let future = u32::try_from(future).map_err(|_| WebgpuErrno::BadHandle)?;
    ctx.data()
        .state
        .webgpu
        .lock()
        .unwrap()
        .drop_waitable(future, WaitableKind::Future)
}

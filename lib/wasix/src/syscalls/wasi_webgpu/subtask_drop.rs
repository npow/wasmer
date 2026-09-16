use crate::state::{WaitableKind, WebgpuErrno};
use crate::syscalls::*;

/// ### `subtask_drop()`
/// Releases subtask handle `subtask`. Errors with [`WebgpuErrno::BadHandle`]
/// if `subtask` names a future instead (see `future_drop`). Safe to call
/// whether or not the subtask has resolved yet -- a pending subtask's
/// background task simply becomes a no-op once it completes (see
/// [`crate::state::WebgpuState::resolve_waitable`]).
#[instrument(level = "trace", skip_all)]
pub fn subtask_drop(ctx: FunctionEnvMut<'_, WasiEnv>, subtask: i32) -> i32 {
    match subtask_drop_inner(ctx, subtask) {
        Ok(()) => WebgpuErrno::Success.to_i32(),
        Err(e) => e.to_i32(),
    }
}

fn subtask_drop_inner(ctx: FunctionEnvMut<'_, WasiEnv>, subtask: i32) -> Result<(), WebgpuErrno> {
    if !ctx.data().capabilities.webgpu_spike.allow {
        return Err(WebgpuErrno::CapabilityDenied);
    }

    let subtask = u32::try_from(subtask).map_err(|_| WebgpuErrno::BadHandle)?;
    ctx.data()
        .state
        .webgpu
        .lock()
        .unwrap()
        .drop_waitable(subtask, WaitableKind::Subtask)
}

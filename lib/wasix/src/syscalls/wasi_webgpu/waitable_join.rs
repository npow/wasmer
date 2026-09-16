use crate::state::WebgpuErrno;
use crate::syscalls::*;

/// ### `waitable_join()`
/// Associates subtask handle `waitable` with waitable set `set`. Phase 1
/// supports exactly one joined waitable per set -- joining a second one
/// before the first is dropped returns [`WebgpuErrno::SetOccupied`] rather
/// than silently overwriting it.
#[instrument(level = "trace", skip_all)]
pub fn waitable_join(ctx: FunctionEnvMut<'_, WasiEnv>, waitable: i32, set: i32) -> i32 {
    match waitable_join_inner(ctx, waitable, set) {
        Ok(()) => WebgpuErrno::Success.to_i32(),
        Err(e) => e.to_i32(),
    }
}

fn waitable_join_inner(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    waitable: i32,
    set: i32,
) -> Result<(), WebgpuErrno> {
    if !ctx.data().capabilities.webgpu_spike.allow {
        return Err(WebgpuErrno::CapabilityDenied);
    }

    let waitable = u32::try_from(waitable).map_err(|_| WebgpuErrno::BadHandle)?;
    let set = u32::try_from(set).map_err(|_| WebgpuErrno::BadHandle)?;

    ctx.data().state.webgpu.lock().unwrap().join(waitable, set)
}

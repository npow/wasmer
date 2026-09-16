use crate::state::WebgpuErrno;
use crate::syscalls::*;

/// ### `waitable_join()`
/// Associates subtask handle `waitable` with waitable set `set`. A set holds
/// up to `MAX_WAITABLES_PER_SET` joined subtasks (Phase 2a's bounded fan-in
/// scope; Phase 1 capped this at exactly one) -- joining past that cap
/// returns [`WebgpuErrno::SetOccupied`] rather than growing the set
/// unboundedly.
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

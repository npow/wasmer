use std::time::Duration;

use crate::state::WebgpuErrno;
use crate::syscalls::*;

/// This bridge has no separate host/guest writer actor for a future, so
/// nothing plays the real spec's `[future-writer]` role -- the only
/// realistic producer of a future's value here is a fake async host/GPU
/// operation. `delay_ms <= 0` uses this default, matching
/// `request_adapter_start`'s own stand-in delay.
const DEFAULT_SIMULATED_DELAY: Duration = Duration::from_millis(50);

/// ### `future_resolve_after()`
/// Fills in a future previously allocated by `future_new`: after `delay_ms`
/// (or the default stand-in delay if `delay_ms <= 0`), resolves `future` to
/// `value` in the background -- the same spawn-a-background-task pattern
/// `request_adapter_start` uses for its own subtask, just decoupled from
/// allocation so it can target a future created earlier. Returns
/// [`WebgpuErrno::BadHandle`] if `future` doesn't exist or isn't a future
/// (e.g. a subtask handle). This call itself never suspends the guest, and
/// is a test/demo stand-in, not part of any real ABI this bridges toward --
/// a real wasi:webgpu bridge's futures would be resolved by native GPU
/// driver callbacks, not a guest-visible host import.
#[instrument(level = "trace", skip_all)]
pub fn future_resolve_after(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    future: i32,
    value: i32,
    delay_ms: i32,
) -> i32 {
    match future_resolve_after_inner(ctx, future, value, delay_ms) {
        Ok(()) => WebgpuErrno::Success.to_i32(),
        Err(e) => e.to_i32(),
    }
}

fn future_resolve_after_inner(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    future: i32,
    value: i32,
    delay_ms: i32,
) -> Result<(), WebgpuErrno> {
    if !ctx.data().capabilities.webgpu_spike.allow {
        return Err(WebgpuErrno::CapabilityDenied);
    }

    let future = u32::try_from(future).map_err(|_| WebgpuErrno::BadHandle)?;
    let value = value as u32;
    let delay = if delay_ms > 0 {
        Duration::from_millis(delay_ms as u64)
    } else {
        DEFAULT_SIMULATED_DELAY
    };

    let env = ctx.data();
    // read_future also rejects a non-Future handle, but only tells us
    // "resolved or not", not the fact that it exists as the right kind --
    // do that check up front so a bad handle fails immediately rather than
    // after a spawned delay.
    env.state.webgpu.lock().unwrap().read_future(future)?;

    let state = env.state.clone();
    let spawn_result = env.tasks().task_shared(Box::new(move || {
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            state.webgpu.lock().unwrap().resolve_waitable(future, value);
        })
    }));
    if spawn_result.is_err() {
        return Err(WebgpuErrno::Unsupported);
    }

    Ok(())
}

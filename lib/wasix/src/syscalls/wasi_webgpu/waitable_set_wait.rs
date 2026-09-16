use wasmer::{AsyncFunctionEnvMut, RuntimeError};

use crate::state::{WaitOutcome, WebgpuErrno};
use crate::syscalls::*;

/// ### `waitable_set_wait()`
/// The one real async host import in this subsystem: suspends the guest's
/// coroutine until `set`'s single joined subtask resolves, then writes its
/// `u32` result payload to `out_event_ptr` and returns
/// [`WebgpuErrno::Success`]. Genuinely suspends (via the same
/// `Function::new_typed_with_env_async` primitive Phase 0 proved) rather
/// than busy-polling; the WasiEnv/store write lock is held only for the two
/// brief synchronous edges (fetching the shared state up front, writing the
/// result at the end), never across the actual wait.
#[instrument(level = "trace", skip(ctx))]
pub async fn waitable_set_wait(
    ctx: AsyncFunctionEnvMut<WasiEnv>,
    set: i32,
    out_event_ptr: i32,
) -> Result<i32, RuntimeError> {
    let Ok(set) = u32::try_from(set) else {
        return Ok(WebgpuErrno::BadHandle.to_i32());
    };

    // Brief synchronous edge #1: check the capability and grab the
    // Arc-shared state, then let the write lock drop immediately -- the
    // suspend below must not hold it.
    let state = {
        let mut write_lock = ctx.write().await;
        let mut sync_env = write_lock.as_function_env_mut();
        let env = sync_env.data();
        if !env.capabilities.webgpu_spike.allow {
            return Ok(WebgpuErrno::CapabilityDenied.to_i32());
        }
        env.state.clone()
    };

    let (subtask, outcome) = match state.webgpu.lock().unwrap().poll_wait(set) {
        Ok(v) => v,
        Err(e) => return Ok(e.to_i32()),
    };

    let payload = match outcome {
        WaitOutcome::Ready(payload) => payload,
        WaitOutcome::Pending(notify) => {
            // The actual suspend point. No WasiEnv/store lock is held here:
            // `notify_one`'s single stored permit means this can't miss the
            // wakeup even if resolution already raced past the check above.
            notify.notified().await;
            match state.webgpu.lock().unwrap().resolved_payload(subtask) {
                Some(payload) => payload,
                None => {
                    // Unreachable in practice: the permit guarantees the
                    // resolving write happened-before this wakeup. Treated
                    // as a handle error rather than panicking.
                    return Ok(WebgpuErrno::BadHandle.to_i32());
                }
            }
        }
    };

    // Brief synchronous edge #2: write the resolved payload back.
    let mut write_lock = ctx.write().await;
    let mut sync_env = write_lock.as_function_env_mut();
    let env = sync_env.data();
    let memory = unsafe { env.memory_view(&sync_env) };
    let out: WasmPtr<u32> = WasmPtr::new(out_event_ptr as u32);
    match out.write(&memory, payload) {
        Ok(()) => Ok(WebgpuErrno::Success.to_i32()),
        Err(_) => Ok(WebgpuErrno::MissingMemory.to_i32()),
    }
}

/// Registered instead of [`waitable_set_wait`] when the engine doesn't
/// support async execution (mirrors `context_switch_not_supported`).
#[instrument(level = "trace", skip(_ctx))]
pub fn waitable_set_wait_not_supported(
    _ctx: FunctionEnvMut<'_, WasiEnv>,
    _set: i32,
    _out_event_ptr: i32,
) -> i32 {
    tracing::warn!(
        "wasi_webgpu_v0::waitable_set_wait is only available on an async-capable engine"
    );
    WebgpuErrno::Unsupported.to_i32()
}

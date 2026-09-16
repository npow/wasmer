use wasmer::{AsyncFunctionEnvMut, RuntimeError};

use crate::state::{WaitOutcome, WebgpuErrno};
use crate::syscalls::*;

/// ### `waitable_set_wait()`
/// The one real async host import in this subsystem: suspends the guest's
/// coroutine until ANY of `set`'s joined subtasks resolves, writes the
/// resolved subtask's handle (first `u32`) and its `u32` result payload
/// (second `u32`) to `out_event_ptr` (8 bytes total -- Phase 1 wrote only a
/// 4-byte payload, but with more than one possible waitable the guest must
/// learn *which* one fired), and returns [`WebgpuErrno::Success`]. That
/// subtask is then removed from `set`'s membership (see
/// [`crate::state::WebgpuState::remove_from_set`]) -- a later
/// `waitable_set_wait` on the same set will not return it again.
///
/// Genuinely suspends (via the same `Function::new_typed_with_env_async`
/// primitive Phase 0 proved) rather than busy-polling. A single wakeup only
/// means *some* joined subtask might have resolved -- another one than the
/// `Notify` that actually fired may have resolved first, or the wakeup may
/// be stale -- so each iteration re-polls all members under the lock rather
/// than trusting which future woke it; the WasiEnv/store write lock is held
/// only for the brief synchronous edges (fetching the shared state up
/// front, writing the result at the end), never across the actual wait.
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

    let (subtask, payload) = loop {
        let outcome = match state.webgpu.lock().unwrap().poll_wait(set) {
            Ok(v) => v,
            Err(e) => return Ok(e.to_i32()),
        };
        match outcome {
            WaitOutcome::Ready(subtask, payload) => break (subtask, payload),
            WaitOutcome::Pending(pending) => {
                // The actual suspend point: wait for whichever of the
                // joined subtasks' `Notify`s fires first. No WasiEnv/store
                // lock is held here. The result is discarded -- which
                // future fired doesn't tell us which subtask is *actually*
                // resolved (see the doc comment above), so we always loop
                // back and re-poll under the lock rather than trust it.
                let waits = pending
                    .into_iter()
                    .map(|(_, notify)| async move { notify.notified().await });
                let _ = futures::future::select_all(waits.map(Box::pin)).await;
            }
        }
    };

    state.webgpu.lock().unwrap().remove_from_set(set, subtask);

    // Brief synchronous edge #2: write the resolved subtask handle and
    // payload back.
    let mut write_lock = ctx.write().await;
    let mut sync_env = write_lock.as_function_env_mut();
    let env = sync_env.data();
    let memory = unsafe { env.memory_view(&sync_env) };
    let out_subtask: WasmPtr<u32> = WasmPtr::new(out_event_ptr as u32);
    let out_payload: WasmPtr<u32> = WasmPtr::new(out_event_ptr as u32 + 4);
    let write_result = out_subtask
        .write(&memory, subtask)
        .and_then(|()| out_payload.write(&memory, payload));
    match write_result {
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

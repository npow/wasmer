use std::time::Duration;

use crate::state::WebgpuErrno;
use crate::syscalls::*;

/// Fake adapter handle a resolved subtask carries. Nothing downstream
/// interprets this value -- there is no real adapter behind it, exactly as
/// in Phase 0.
const FAKE_ADAPTER_ID: u32 = 1;
/// Default stand-in delay for a real "wait on the native GPU driver" host
/// operation, used by callers that pass `delay_ms <= 0`.
const DEFAULT_SIMULATED_DELAY: Duration = Duration::from_millis(50);

/// ### `request_adapter_start()`
/// Starts the fake "request an adapter" operation without blocking: returns
/// a subtask handle immediately, and resolves it in the background after
/// `delay_ms` (or [`DEFAULT_SIMULATED_DELAY`] if `delay_ms <= 0`). The guest
/// must `waitable_join` the subtask into a waitable set and
/// `waitable_set_wait` on that set to actually observe the result -- this
/// call itself never suspends the guest.
///
/// `delay_ms` is a test/demo knob, not something a real `wasi:webgpu`
/// `request-adapter` would expose -- it exists so a caller (a test, or this
/// crate's own guest-driven demos) can stagger several fake operations'
/// completion order to exercise Phase 2a's "whichever joined subtask
/// resolves first" semantics deterministically, rather than racing several
/// identically-timed background tasks.
#[instrument(level = "trace", skip_all)]
pub fn request_adapter_start(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    out_subtask_ptr: i32,
    delay_ms: i32,
) -> i32 {
    match request_adapter_start_inner(ctx, out_subtask_ptr, delay_ms) {
        Ok(()) => WebgpuErrno::Success.to_i32(),
        Err(e) => e.to_i32(),
    }
}

fn request_adapter_start_inner(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    out_subtask_ptr: i32,
    delay_ms: i32,
) -> Result<(), WebgpuErrno> {
    if !ctx.data().capabilities.webgpu_spike.allow {
        return Err(WebgpuErrno::CapabilityDenied);
    }

    let delay = if delay_ms > 0 {
        Duration::from_millis(delay_ms as u64)
    } else {
        DEFAULT_SIMULATED_DELAY
    };

    let env = ctx.data();
    let handle = env
        .state
        .webgpu
        .lock()
        .unwrap()
        .insert_pending(crate::state::WaitableKind::Subtask)?;

    // Spawned onto the shared task pool, not run inline: this is the "start"
    // half of start/wait, so it must return to the guest immediately.
    let state = env.state.clone();
    let spawn_result = env.tasks().task_shared(Box::new(move || {
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            state
                .webgpu
                .lock()
                .unwrap()
                .resolve_waitable(handle, FAKE_ADAPTER_ID);
        })
    }));
    if spawn_result.is_err() {
        let _ = env
            .state
            .webgpu
            .lock()
            .unwrap()
            .drop_waitable(handle, crate::state::WaitableKind::Subtask);
        return Err(WebgpuErrno::Unsupported);
    }

    let memory = unsafe { env.memory_view(&ctx) };
    let out: WasmPtr<u32> = WasmPtr::new(out_subtask_ptr as u32);
    out.write(&memory, handle)
        .map_err(|_| WebgpuErrno::MissingMemory)
}

use std::time::Duration;

use crate::state::WebgpuErrno;
use crate::syscalls::*;

/// Fake adapter handle a resolved subtask carries. Nothing downstream
/// interprets this value -- there is no real adapter behind it, exactly as
/// in Phase 0.
const FAKE_ADAPTER_ID: u32 = 1;
/// Stand-in for a real "wait on the native GPU driver" host operation.
const SIMULATED_DELAY: Duration = Duration::from_millis(50);

/// ### `request_adapter_start()`
/// Starts the fake "request an adapter" operation without blocking: returns
/// a subtask handle immediately, and resolves it in the background after
/// [`SIMULATED_DELAY`]. The guest must `waitable_join` the subtask into a
/// waitable set and `waitable_set_wait` on that set to actually observe the
/// result -- this call itself never suspends the guest.
#[instrument(level = "trace", skip_all)]
pub fn request_adapter_start(ctx: FunctionEnvMut<'_, WasiEnv>, out_subtask_ptr: i32) -> i32 {
    match request_adapter_start_inner(ctx, out_subtask_ptr) {
        Ok(()) => WebgpuErrno::Success.to_i32(),
        Err(e) => e.to_i32(),
    }
}

fn request_adapter_start_inner(
    ctx: FunctionEnvMut<'_, WasiEnv>,
    out_subtask_ptr: i32,
) -> Result<(), WebgpuErrno> {
    if !ctx.data().capabilities.webgpu_spike.allow {
        return Err(WebgpuErrno::CapabilityDenied);
    }

    let env = ctx.data();
    let (handle, notify) = env.state.webgpu.lock().unwrap().insert_pending_subtask()?;

    // Spawned onto the shared task pool, not run inline: this is the "start"
    // half of start/wait, so it must return to the guest immediately.
    let state = env.state.clone();
    let spawn_result = env.tasks().task_shared(Box::new(move || {
        Box::pin(async move {
            tokio::time::sleep(SIMULATED_DELAY).await;
            state
                .webgpu
                .lock()
                .unwrap()
                .resolve_subtask(handle, FAKE_ADAPTER_ID);
            notify.notify_one();
        })
    }));
    if spawn_result.is_err() {
        let _ = env.state.webgpu.lock().unwrap().drop_subtask(handle);
        return Err(WebgpuErrno::Unsupported);
    }

    let memory = unsafe { env.memory_view(&ctx) };
    let out: WasmPtr<u32> = WasmPtr::new(out_subtask_ptr as u32);
    out.write(&memory, handle)
        .map_err(|_| WebgpuErrno::MissingMemory)
}

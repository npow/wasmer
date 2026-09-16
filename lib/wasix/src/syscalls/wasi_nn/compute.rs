use crate::syscalls::*;
use wasmer_wasi_nn::{NnErrno, NnExecutionContext};

/// ### `compute()`
/// Runs inference on execution context `context`.
#[instrument(level = "trace", skip_all)]
pub fn nn_compute(ctx: FunctionEnvMut<'_, WasiEnv>, context: u32) -> u32 {
    let env = ctx.data();
    let mut nn = env.state.nn.lock().unwrap();
    let exec_ctx = match nn.context_mut(context) {
        Some(exec_ctx) => exec_ctx,
        None => return NnErrno::NotFound.to_u32(),
    };

    match run_compute(exec_ctx) {
        Ok(()) => NnErrno::Success.to_u32(),
        Err(err) => err.to_u32(),
    }
}

/// Runs `.compute()` -- real inference, which blocks a native thread doing CPU
/// matmuls or waiting on a CUDA call -- without stalling whatever tokio
/// executor thread this host function happens to run on.
///
/// `nn_compute` is registered as a plain sync host function (see
/// `wasi_ephemeral_nn_exports` in `lib.rs`, which is not ours to edit), so it
/// can't suspend the guest instance the way `sock_recv`'s asyncify-based
/// `__sock_asyncify` does to hand blocking work to
/// `VirtualTaskManager::task_dedicated` -- and `task_dedicated` needs a
/// `Send + 'static` closure anyway, which the `&mut dyn NnExecutionContext`
/// borrowed out of the locked `NnState` is not. `block_in_place` runs the call
/// in place (so the borrow is fine, no channel/thread hand-off needed) while
/// telling tokio this worker is about to block, so it can move other queued
/// work onto a spare worker thread.
///
/// `block_in_place` panics outside a multi-threaded runtime (no other worker to
/// hand off to), so this only takes that path when one is confirmed current;
/// it falls back to a direct, in-place call otherwise (no tokio runtime
/// entered, or a single-threaded `current_thread` one).
///
/// This is the pragmatic fix, not the ideal one: registering `compute` with
/// `Function::new_typed_with_env_async` and using
/// `VirtualTaskManager::task_dedicated` proper would let the guest instance
/// yield instead of blocking a native thread, but that changes `compute`'s
/// registration in `lib.rs`, which is outside this change's file set.
fn run_compute(exec_ctx: &mut dyn NnExecutionContext) -> Result<(), NnErrno> {
    #[cfg(feature = "sys-thread")]
    {
        use tokio::runtime::RuntimeFlavor;
        let on_multi_thread = tokio::runtime::Handle::try_current()
            .is_ok_and(|h| h.runtime_flavor() == RuntimeFlavor::MultiThread);
        if on_multi_thread {
            return tokio::task::block_in_place(|| exec_ctx.compute());
        }
    }

    exec_ctx.compute()
}

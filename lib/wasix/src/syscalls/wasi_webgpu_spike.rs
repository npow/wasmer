//! Phase-0 spike: a single, deliberately fake async wasix host import
//! proving Wasmer can suspend a guest's execution mid-call and resume it
//! later with data, using the engine's *existing* JSPI-style coroutine
//! primitive (see `wasmer::AsyncFunctionEnvMut` /
//! `lib/api/src/backend/sys/async_runtime.rs`) -- the same one
//! `context_switch` (`syscalls/wasix/context_switch.rs`) already rides in
//! production.
//!
//! This is NOT wasi:webgpu. It implements none of that WIT interface, its
//! canonical ABI, or the Component Model. It exists only to answer one
//! question: can a real `.wasm` guest call an async wasix host import, have
//! the call suspend on a real `.await`, and resume correctly with the right
//! data? If a real wasi:webgpu bridge is ever built, `request-adapter` would
//! still need the actual canonical-ABI lowering (`task.return`,
//! `waitable-set.*`, ...) this deliberately skips.
//!
//! Gated by this crate's `wasi-webgpu-spike` feature. Never registered
//! unless that feature is enabled.

use crate::WasiEnv;
use std::time::Duration;
use tracing::instrument;
use wasmer::{AsyncFunctionEnvMut, FunctionEnvMut, RuntimeError, WasmPtr};

/// Fake adapter handle written back into guest memory on "success". Nothing
/// downstream interprets this value -- there is no real adapter behind it.
const FAKE_ADAPTER_ID: u32 = 1;

/// Guest calls this expecting a real `request-adapter`-shaped async host
/// call. The `.await` below is what actually suspends the guest's coroutine
/// and lets other host work run before it resumes -- proving that
/// suspend/resume round trip is the entire point of this spike, not the
/// fake delay or the fake adapter id themselves.
#[instrument(level = "trace", skip(ctx))]
pub async fn request_adapter_spike(
    ctx: AsyncFunctionEnvMut<WasiEnv>,
    out_ptr: i32,
) -> Result<i32, RuntimeError> {
    // Stand-in for a real "wait on the native GPU driver" host operation.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut write_lock = ctx.write().await;
    let mut sync_env = write_lock.as_function_env_mut();
    let env = sync_env.data();
    let memory = unsafe { env.memory_view(&sync_env) };

    let out: WasmPtr<u32> = WasmPtr::new(out_ptr as u32);
    match out.write(&memory, FAKE_ADAPTER_ID) {
        Ok(()) => Ok(0),
        Err(_) => Ok(-1),
    }
}

/// Registered instead of [`request_adapter_spike`] when the engine doesn't
/// support async execution (mirrors `context_switch_not_supported`).
#[instrument(level = "trace", skip(_ctx))]
pub fn request_adapter_spike_not_supported(
    _ctx: FunctionEnvMut<'_, WasiEnv>,
    _out_ptr: i32,
) -> i32 {
    tracing::warn!(
        "wasi_webgpu_v0::request_adapter_spike is only available on an async-capable engine"
    );
    -2
}

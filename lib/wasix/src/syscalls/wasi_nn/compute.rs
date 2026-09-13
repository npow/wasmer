use crate::syscalls::*;

/// ### `compute()`
/// Runs inference on execution context `context`.
///
/// TODO(wasi-nn): body. Real inference blocks a native thread (CPU or CUDA);
/// do not call the backend directly on this async runtime thread. Follow
/// whatever pattern `fd_read`/`sock_recv` use to hand blocking work off to
/// `WasiEnv::tasks()` / the runtime's task manager, not a raw
/// `tokio::task::spawn_blocking`.
#[instrument(level = "trace", skip_all)]
pub fn nn_compute(mut _ctx: FunctionEnvMut<'_, WasiEnv>, _context: u32) -> u32 {
    wasmer_wasi_nn::NnErrno::UnsupportedOperation.to_u32()
}

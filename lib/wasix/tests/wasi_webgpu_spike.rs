#![cfg(feature = "wasi-webgpu-spike")]

//! Phase-0 spike test: proves a real `.wasm` guest can call an *async* wasix
//! host import (`wasi_webgpu_v0::request_adapter_spike`), have that call
//! genuinely suspend the guest's coroutine on a real `.await`, and resume
//! correctly with data -- using Wasmer's existing JSPI-style
//! `Function::new_typed_with_env_async`/`AsyncFunctionEnvMut` primitive (the
//! same one `context_switch` already rides in production). See
//! `lib/wasix/src/syscalls/wasi_webgpu_spike.rs` for what this deliberately
//! does and does not prove -- it is not a wasi:webgpu implementation.

use std::time::{Duration, Instant};

use wasmer::{Module, Store, Value};
use wasmer_wasix::WasiEnv;

const WASI_WEBGPU_SPIKE_WAT: &str = r#"
(module
  (import "wasi_webgpu_v0" "request_adapter_spike"
    (func $request_adapter_spike (param i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "do_request_adapter_spike") (param $out_ptr i32) (result i32)
    local.get $out_ptr
    call $request_adapter_spike))
"#;

const OUT_PTR: i32 = 0;
const FAKE_ADAPTER_ID: u32 = 1;
/// Must match `request_adapter_spike`'s simulated host delay.
const SIMULATED_DELAY: Duration = Duration::from_millis(50);

#[test]
fn request_adapter_spike_suspends_and_resumes_through_a_real_guest() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let mut store = Store::default();
    let module = Module::new(&store, WASI_WEBGPU_SPIKE_WAT).expect("compile guest WAT module");

    let builder = WasiEnv::builder("wasi-webgpu-spike-test").engine(store.engine().clone());
    let (instance, wasi_env) = builder
        .instantiate(module, &mut store)
        .expect("instantiate guest module with wasi_webgpu_v0 imports");

    let do_request_adapter_spike = instance
        .exports
        .get_function("do_request_adapter_spike")
        .expect("do_request_adapter_spike export")
        .clone();

    // Move the store into its async form: `Function::call_async` is how a
    // caller drives a wasm call that may suspend on an async host import.
    let store_async = store.into_async();

    let started = Instant::now();
    let result = runtime
        .block_on(do_request_adapter_spike.call_async(&store_async, vec![Value::I32(OUT_PTR)]));
    let elapsed = started.elapsed();

    let mut store = store_async
        .into_store()
        .expect("store_async should be uniquely owned and unlocked once the call has completed");

    let return_code = match result.expect("call do_request_adapter_spike")[0] {
        Value::I32(v) => v,
        ref other => panic!("unexpected return value: {other:?}"),
    };
    assert_eq!(
        return_code, 0,
        "request_adapter_spike should report success"
    );

    // Proves the guest's coroutine genuinely suspended on the host's
    // `.await` rather than the whole call returning synchronously: real
    // wall-clock time elapsed across the call, matching the simulated delay.
    assert!(
        elapsed >= SIMULATED_DELAY,
        "expected the call to block for at least {SIMULATED_DELAY:?} while the guest was \
         suspended waiting on the host future; only took {elapsed:?}"
    );

    // Proves it resumed with the right data: the fake adapter id, written
    // into guest memory only *after* the suspend point.
    let memory = instance
        .exports
        .get_memory("memory")
        .expect("guest exports memory");
    let mut adapter_id_bytes = [0u8; 4];
    memory
        .view(&store)
        .read(OUT_PTR as u64, &mut adapter_id_bytes)
        .expect("read fake adapter id back from guest memory");
    assert_eq!(
        u32::from_le_bytes(adapter_id_bytes),
        FAKE_ADAPTER_ID,
        "guest memory should contain the fake adapter id written after resume"
    );

    wasi_env.on_exit(&mut store, None);
}

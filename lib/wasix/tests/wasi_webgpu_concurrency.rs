#![cfg(feature = "wasi-webgpu-concurrency")]

//! Phase 1 test: proves the graduated `wasi_webgpu_v0` subset -- start a
//! subtask non-blockingly, join it to a waitable set, and only actually
//! suspend the guest in `waitable_set_wait` -- works end to end through a
//! real `.wasm` guest, and that Phase 1's one-waitable-per-set boundary is
//! enforced rather than silently ignored.
//!
//! Supersedes Phase 0's `tests/wasi_webgpu_spike.rs` (now removed): that
//! file tested a single fake `request_adapter_spike` import that both
//! started and waited in one call. This tests the real start/join/wait/drop
//! shape a Component Model async export actually needs.

use std::time::{Duration, Instant};

use wasmer::{Module, Store, Value};
use wasmer_wasix::WasiEnv;

const WASI_WEBGPU_V0_WAT: &str = r#"
(module
  (import "wasi_webgpu_v0" "request_adapter_start"
    (func $request_adapter_start (param i32) (result i32)))
  (import "wasi_webgpu_v0" "waitable_set_new"
    (func $waitable_set_new (param i32) (result i32)))
  (import "wasi_webgpu_v0" "waitable_join"
    (func $waitable_join (param i32 i32) (result i32)))
  (import "wasi_webgpu_v0" "waitable_set_wait"
    (func $waitable_set_wait (param i32 i32) (result i32)))
  (import "wasi_webgpu_v0" "subtask_drop"
    (func $subtask_drop (param i32) (result i32)))
  (memory (export "memory") 1)

  (func (export "do_request_adapter_start") (param $out_subtask_ptr i32) (result i32)
    local.get $out_subtask_ptr
    call $request_adapter_start)

  (func (export "do_waitable_set_new") (param $out_set_ptr i32) (result i32)
    local.get $out_set_ptr
    call $waitable_set_new)

  (func (export "do_waitable_join") (param $waitable i32) (param $set i32) (result i32)
    local.get $waitable
    local.get $set
    call $waitable_join)

  (func (export "do_waitable_set_wait") (param $set i32) (param $out_event_ptr i32) (result i32)
    local.get $set
    local.get $out_event_ptr
    call $waitable_set_wait)

  (func (export "do_subtask_drop") (param $subtask i32) (result i32)
    local.get $subtask
    call $subtask_drop))
"#;

const SUBTASK_OUT_PTR: i32 = 0;
const SET_OUT_PTR: i32 = 4;
const EVENT_OUT_PTR: i32 = 8;
const SECOND_SET_OUT_PTR: i32 = 12;
const FAKE_ADAPTER_ID: u32 = 1;
/// Must match `request_adapter_start`'s simulated host delay.
const SIMULATED_DELAY: Duration = Duration::from_millis(50);

fn new_instance() -> (
    wasmer::Instance,
    Store,
    wasmer_wasix::WasiFunctionEnv,
    wasmer::Memory,
) {
    let mut store = Store::default();
    let module = Module::new(&store, WASI_WEBGPU_V0_WAT).expect("compile guest WAT module");

    let mut builder =
        WasiEnv::builder("wasi-webgpu-concurrency-test").engine(store.engine().clone());
    builder.capabilities_mut().webgpu_spike.allow = true;

    let (instance, wasi_env) = builder
        .instantiate(module, &mut store)
        .expect("instantiate guest module with wasi_webgpu_v0 imports");
    let memory = instance
        .exports
        .get_memory("memory")
        .expect("guest exports memory")
        .clone();
    (instance, store, wasi_env, memory)
}

fn read_u32(memory: &wasmer::Memory, store: &Store, ptr: i32) -> u32 {
    let mut bytes = [0u8; 4];
    memory
        .view(store)
        .read(ptr as u64, &mut bytes)
        .expect("read u32 from guest memory");
    u32::from_le_bytes(bytes)
}

#[test]
fn waitable_set_wait_suspends_and_resumes_through_a_real_guest() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let (instance, store, wasi_env, memory) = new_instance();

    let do_request_adapter_start = instance
        .exports
        .get_function("do_request_adapter_start")
        .unwrap()
        .clone();
    let do_waitable_set_new = instance
        .exports
        .get_function("do_waitable_set_new")
        .unwrap()
        .clone();
    let do_waitable_join = instance
        .exports
        .get_function("do_waitable_join")
        .unwrap()
        .clone();
    let do_waitable_set_wait = instance
        .exports
        .get_function("do_waitable_set_wait")
        .unwrap()
        .clone();
    let do_subtask_drop = instance
        .exports
        .get_function("do_subtask_drop")
        .unwrap()
        .clone();

    // request_adapter_start and waitable_set_new/join are plain sync host
    // imports -- drive them with a normal, non-async store.
    let mut store = store;
    let rc = do_request_adapter_start
        .call(&mut store, &[Value::I32(SUBTASK_OUT_PTR)])
        .expect("call do_request_adapter_start");
    assert_eq!(rc[0], Value::I32(0), "request_adapter_start should succeed");
    let subtask = read_u32(&memory, &store, SUBTASK_OUT_PTR);

    let rc = do_waitable_set_new
        .call(&mut store, &[Value::I32(SET_OUT_PTR)])
        .expect("call do_waitable_set_new");
    assert_eq!(rc[0], Value::I32(0), "waitable_set_new should succeed");
    let set = read_u32(&memory, &store, SET_OUT_PTR);

    let rc = do_waitable_join
        .call(
            &mut store,
            &[Value::I32(subtask as i32), Value::I32(set as i32)],
        )
        .expect("call do_waitable_join");
    assert_eq!(rc[0], Value::I32(0), "waitable_join should succeed");

    // waitable_set_wait is the one real async host import: switch to the
    // async calling convention to drive it.
    let store_async = store.into_async();
    let started = Instant::now();
    let result = runtime.block_on(do_waitable_set_wait.call_async(
        &store_async,
        vec![Value::I32(set as i32), Value::I32(EVENT_OUT_PTR)],
    ));
    let elapsed = started.elapsed();
    let mut store = store_async
        .into_store()
        .expect("store_async should be uniquely owned and unlocked once the call has completed");

    let rc = result.expect("call do_waitable_set_wait");
    assert_eq!(rc[0], Value::I32(0), "waitable_set_wait should succeed");

    // Proves the guest's coroutine genuinely suspended rather than the call
    // returning synchronously: real wall-clock time elapsed, matching the
    // simulated delay -- exactly Phase 0's verification technique.
    assert!(
        elapsed >= SIMULATED_DELAY,
        "expected waitable_set_wait to block for at least {SIMULATED_DELAY:?} while the guest \
         was suspended; only took {elapsed:?}"
    );

    let event = read_u32(&memory, &store, EVENT_OUT_PTR);
    assert_eq!(
        event, FAKE_ADAPTER_ID,
        "resolved event payload should be the fake adapter id"
    );

    let rc = do_subtask_drop
        .call(&mut store, &[Value::I32(subtask as i32)])
        .expect("call do_subtask_drop");
    assert_eq!(rc[0], Value::I32(0), "subtask_drop should succeed");

    wasi_env.on_exit(&mut store, None);
}

#[test]
fn waitable_join_rejects_a_second_waitable_on_an_occupied_set() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let (instance, mut store, wasi_env, memory) = new_instance();

    let do_request_adapter_start = instance
        .exports
        .get_function("do_request_adapter_start")
        .unwrap()
        .clone();
    let do_waitable_set_new = instance
        .exports
        .get_function("do_waitable_set_new")
        .unwrap()
        .clone();
    let do_waitable_join = instance
        .exports
        .get_function("do_waitable_join")
        .unwrap()
        .clone();

    // Two independent subtasks...
    do_request_adapter_start
        .call(&mut store, &[Value::I32(SUBTASK_OUT_PTR)])
        .unwrap();
    let subtask_a = read_u32(&memory, &store, SUBTASK_OUT_PTR);
    do_request_adapter_start
        .call(&mut store, &[Value::I32(SECOND_SET_OUT_PTR)])
        .unwrap();
    let subtask_b = read_u32(&memory, &store, SECOND_SET_OUT_PTR);

    // ...and one set.
    do_waitable_set_new
        .call(&mut store, &[Value::I32(SET_OUT_PTR)])
        .unwrap();
    let set = read_u32(&memory, &store, SET_OUT_PTR);

    let rc = do_waitable_join
        .call(
            &mut store,
            &[Value::I32(subtask_a as i32), Value::I32(set as i32)],
        )
        .expect("call do_waitable_join (first)");
    assert_eq!(rc[0], Value::I32(0), "first join should succeed");

    // Joining a second waitable to the same, already-occupied set must be
    // rejected, not silently overwrite the first join or panic.
    let rc = do_waitable_join
        .call(
            &mut store,
            &[Value::I32(subtask_b as i32), Value::I32(set as i32)],
        )
        .expect("call do_waitable_join (second)");
    assert_eq!(
        rc[0],
        Value::I32(-5),
        "second join to an occupied set should return SetOccupied (-5), not succeed"
    );

    wasi_env.on_exit(&mut store, None);
}

#![cfg(feature = "wasi-webgpu-concurrency")]

//! `future.*` test: proves `future_new`/`future_resolve_after`/`future_read`/
//! `future_drop` work end to end through a real `.wasm` guest, in both the
//! order-enforced synchronous path (value already there when read) and the
//! suspend-based path (guest reads first, gets `Blocked`, joins the future
//! into a waitable set, and genuinely suspends on `waitable_set_wait` until
//! it resolves) -- reusing Phase 2a's exact waitable-set machinery, no new
//! suspension primitive. Also proves a subtask and a future can share one
//! waitable set and still resolve in the correct order, the actual risk of
//! generalizing `WebgpuState`'s waitable table to cover both kinds.

use std::time::{Duration, Instant};

use wasmer::{Module, Store, Value};
use wasmer_wasix::WasiEnv;

const WASI_WEBGPU_V0_WAT: &str = r#"
(module
  (import "wasi_webgpu_v0" "request_adapter_start"
    (func $request_adapter_start (param i32 i32) (result i32)))
  (import "wasi_webgpu_v0" "waitable_set_new"
    (func $waitable_set_new (param i32) (result i32)))
  (import "wasi_webgpu_v0" "waitable_join"
    (func $waitable_join (param i32 i32) (result i32)))
  (import "wasi_webgpu_v0" "waitable_set_wait"
    (func $waitable_set_wait (param i32 i32) (result i32)))
  (import "wasi_webgpu_v0" "subtask_drop"
    (func $subtask_drop (param i32) (result i32)))
  (import "wasi_webgpu_v0" "future_new"
    (func $future_new (param i32) (result i32)))
  (import "wasi_webgpu_v0" "future_resolve_after"
    (func $future_resolve_after (param i32 i32 i32) (result i32)))
  (import "wasi_webgpu_v0" "future_read"
    (func $future_read (param i32 i32) (result i32)))
  (import "wasi_webgpu_v0" "future_drop"
    (func $future_drop (param i32) (result i32)))
  (memory (export "memory") 1)

  (func (export "do_request_adapter_start")
        (param $out_subtask_ptr i32) (param $delay_ms i32) (result i32)
    local.get $out_subtask_ptr
    local.get $delay_ms
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
    call $subtask_drop)

  (func (export "do_future_new") (param $out_future_ptr i32) (result i32)
    local.get $out_future_ptr
    call $future_new)

  (func (export "do_future_resolve_after")
        (param $future i32) (param $value i32) (param $delay_ms i32) (result i32)
    local.get $future
    local.get $value
    local.get $delay_ms
    call $future_resolve_after)

  (func (export "do_future_read") (param $future i32) (param $out_value_ptr i32) (result i32)
    local.get $future
    local.get $out_value_ptr
    call $future_read)

  (func (export "do_future_drop") (param $future i32) (result i32)
    local.get $future
    call $future_drop))
"#;

const SUBTASK_OUT_PTR: i32 = 0;
const SET_OUT_PTR: i32 = 4;
/// 8 bytes: resolved waitable handle (first `u32`), then its payload
/// (second `u32`).
const EVENT_OUT_PTR: i32 = 8;
const FUTURE_OUT_PTR: i32 = 16;
const FUTURE_VALUE_OUT_PTR: i32 = 20;
/// Must match `request_adapter_start`'s/`future_resolve_after`'s default
/// simulated host delay (used whenever a test passes `delay_ms <= 0`).
const DEFAULT_SIMULATED_DELAY: Duration = Duration::from_millis(50);
const JITTER_TOLERANCE: Duration = Duration::from_millis(5);
const FAKE_ADAPTER_ID: u32 = 1;
const BLOCKED: i32 = -8;

struct Guest {
    instance: wasmer::Instance,
    store: Store,
    wasi_env: wasmer_wasix::WasiFunctionEnv,
    memory: wasmer::Memory,
}

fn new_guest() -> Guest {
    let mut store = Store::default();
    let module = Module::new(&store, WASI_WEBGPU_V0_WAT).expect("compile guest WAT module");

    let mut builder = WasiEnv::builder("wasi-webgpu-future-test").engine(store.engine().clone());
    builder.capabilities_mut().webgpu_spike.allow = true;

    let (instance, wasi_env) = builder
        .instantiate(module, &mut store)
        .expect("instantiate guest module with wasi_webgpu_v0 imports");
    let memory = instance
        .exports
        .get_memory("memory")
        .expect("guest exports memory")
        .clone();
    Guest {
        instance,
        store,
        wasi_env,
        memory,
    }
}

fn read_u32(memory: &wasmer::Memory, store: &Store, ptr: i32) -> u32 {
    let mut bytes = [0u8; 4];
    memory
        .view(store)
        .read(ptr as u64, &mut bytes)
        .expect("read u32 from guest memory");
    u32::from_le_bytes(bytes)
}

fn call1(guest: &mut Guest, name: &str, args: &[Value]) -> i32 {
    let f = guest.instance.exports.get_function(name).unwrap().clone();
    let rc = f
        .call(&mut guest.store, args)
        .unwrap_or_else(|e| panic!("call {name}: {e}"));
    let Value::I32(rc) = rc[0] else {
        panic!("{name} should return an i32");
    };
    rc
}

fn request_adapter_start(guest: &mut Guest, delay_ms: i32) -> u32 {
    let rc = call1(
        guest,
        "do_request_adapter_start",
        &[Value::I32(SUBTASK_OUT_PTR), Value::I32(delay_ms)],
    );
    assert_eq!(rc, 0, "request_adapter_start should succeed");
    read_u32(&guest.memory, &guest.store, SUBTASK_OUT_PTR)
}

fn waitable_set_new(guest: &mut Guest) -> u32 {
    let rc = call1(guest, "do_waitable_set_new", &[Value::I32(SET_OUT_PTR)]);
    assert_eq!(rc, 0, "waitable_set_new should succeed");
    read_u32(&guest.memory, &guest.store, SET_OUT_PTR)
}

fn waitable_join(guest: &mut Guest, waitable: u32, set: u32) -> i32 {
    call1(
        guest,
        "do_waitable_join",
        &[Value::I32(waitable as i32), Value::I32(set as i32)],
    )
}

fn future_new(guest: &mut Guest) -> u32 {
    let rc = call1(guest, "do_future_new", &[Value::I32(FUTURE_OUT_PTR)]);
    assert_eq!(rc, 0, "future_new should succeed");
    read_u32(&guest.memory, &guest.store, FUTURE_OUT_PTR)
}

fn future_resolve_after(guest: &mut Guest, future: u32, value: i32, delay_ms: i32) -> i32 {
    call1(
        guest,
        "do_future_resolve_after",
        &[
            Value::I32(future as i32),
            Value::I32(value),
            Value::I32(delay_ms),
        ],
    )
}

fn future_read_call(guest: &mut Guest, future: u32) -> i32 {
    call1(
        guest,
        "do_future_read",
        &[Value::I32(future as i32), Value::I32(FUTURE_VALUE_OUT_PTR)],
    )
}

fn future_drop(guest: &mut Guest, future: u32) -> i32 {
    call1(guest, "do_future_drop", &[Value::I32(future as i32)])
}

#[test]
fn future_read_returns_the_value_immediately_once_already_resolved() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let mut guest = new_guest();

    let future = future_new(&mut guest);
    // A short, real delay -- resolved well before this thread's subsequent
    // sleep, so the read below hits the already-resolved fast path
    // deterministically rather than racing the background task.
    assert_eq!(future_resolve_after(&mut guest, future, 42, 1), 0);
    std::thread::sleep(Duration::from_millis(30));

    let rc = future_read_call(&mut guest, future);
    assert_eq!(
        rc, 0,
        "future_read should succeed once the future has resolved"
    );
    let value = read_u32(&guest.memory, &guest.store, FUTURE_VALUE_OUT_PTR);
    assert_eq!(value, 42);

    assert_eq!(future_drop(&mut guest, future), 0);
    let mut store = guest.store;
    guest.wasi_env.on_exit(&mut store, None);
}

#[test]
fn future_read_blocks_then_resolves_via_waitable_set_wait() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let mut guest = new_guest();

    let future = future_new(&mut guest);
    assert_eq!(
        future_resolve_after(&mut guest, future, 7, 0 /* default delay */),
        0
    );

    // Read immediately, before the background task has had a chance to run
    // -- must report Blocked, not a stale/garbage value.
    assert_eq!(
        future_read_call(&mut guest, future),
        BLOCKED,
        "future_read on an unresolved future should return Blocked"
    );

    let set = waitable_set_new(&mut guest);
    assert_eq!(waitable_join(&mut guest, future, set), 0);

    let do_waitable_set_wait = guest
        .instance
        .exports
        .get_function("do_waitable_set_wait")
        .unwrap()
        .clone();
    let store_async = guest.store.into_async();
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

    assert!(
        elapsed + JITTER_TOLERANCE >= DEFAULT_SIMULATED_DELAY,
        "expected waitable_set_wait to genuinely suspend for at least {DEFAULT_SIMULATED_DELAY:?}; \
         only took {elapsed:?}"
    );

    let resolved_handle = read_u32(&guest.memory, &store, EVENT_OUT_PTR);
    let payload = read_u32(&guest.memory, &store, EVENT_OUT_PTR + 4);
    assert_eq!(
        resolved_handle, future,
        "the event should name the future's own handle"
    );
    assert_eq!(payload, 7);

    guest.wasi_env.on_exit(&mut store, None);
}

#[test]
fn a_subtask_and_a_future_can_share_one_waitable_set_and_resolve_in_order() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let mut guest = new_guest();
    let set = waitable_set_new(&mut guest);

    // The future resolves faster (30ms) than the subtask (100ms) -- if the
    // generalized waitable table mishandled kind-mixing (e.g. only ever
    // scanning one kind, or always preferring one over the other), this
    // would return the wrong one first.
    let future = future_new(&mut guest);
    assert_eq!(future_resolve_after(&mut guest, future, 99, 30), 0);
    assert_eq!(waitable_join(&mut guest, future, set), 0);

    let subtask = request_adapter_start(&mut guest, 100);
    assert_eq!(waitable_join(&mut guest, subtask, set), 0);

    let do_waitable_set_wait = guest
        .instance
        .exports
        .get_function("do_waitable_set_wait")
        .unwrap()
        .clone();

    let mut resolution_order = Vec::new();
    let mut store = guest.store;
    for _ in 0..2 {
        let store_async = store.into_async();
        let result = runtime.block_on(do_waitable_set_wait.call_async(
            &store_async,
            vec![Value::I32(set as i32), Value::I32(EVENT_OUT_PTR)],
        ));
        store = store_async.into_store().expect(
            "store_async should be uniquely owned and unlocked once the call has completed",
        );
        let rc = result.expect("call do_waitable_set_wait");
        assert_eq!(rc[0], Value::I32(0));
        resolution_order.push(read_u32(&guest.memory, &store, EVENT_OUT_PTR));
    }

    assert_eq!(
        resolution_order,
        vec![future, subtask],
        "the future (30ms) should resolve before the subtask (100ms), regardless of kind"
    );

    let payload_of_first = read_u32(&guest.memory, &store, EVENT_OUT_PTR + 4);
    // This reads whatever is currently at EVENT_OUT_PTR+4, which after the
    // loop above holds the *second* wait's payload (the subtask's fake
    // adapter id) -- assert that instead of the stale first value.
    assert_eq!(payload_of_first, FAKE_ADAPTER_ID);

    guest.wasi_env.on_exit(&mut store, None);
}

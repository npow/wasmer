#![cfg(feature = "wasi-webgpu-concurrency")]

//! Phase 2a test: proves the graduated `wasi_webgpu_v0` subset -- start a
//! subtask non-blockingly, join up to `MAX_WAITABLES_PER_SET` of them into a
//! single waitable set, and have `waitable_set_wait` genuinely suspend the
//! guest and return whichever one resolves *first* (not join order, not
//! handle-number order) -- works end to end through a real `.wasm` guest,
//! and that the set's capacity cap is enforced rather than silently ignored.
//!
//! Supersedes Phase 0's `tests/wasi_webgpu_spike.rs` (removed then): that
//! file tested a single fake `request_adapter_spike` import that both
//! started and waited in one call. This tests the real start/join/wait/drop
//! shape a Component Model async export actually needs.

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
    call $subtask_drop))
"#;

const SUBTASK_OUT_PTR: i32 = 0;
const SET_OUT_PTR: i32 = 4;
/// 8 bytes: resolved subtask handle (first `u32`), then its payload (second
/// `u32`) -- Phase 2a's wider event record (Phase 1 wrote only a 4-byte
/// payload, since there was never more than one possible waitable).
const EVENT_OUT_PTR: i32 = 8;
const FAKE_ADAPTER_ID: u32 = 1;
/// Must match `request_adapter_start`'s default simulated host delay
/// (used whenever a test passes `delay_ms <= 0`).
const DEFAULT_SIMULATED_DELAY: Duration = Duration::from_millis(50);
/// Must match `MAX_WAITABLES_PER_SET` in `state/webgpu.rs`.
const MAX_WAITABLES_PER_SET: usize = 8;

struct Guest {
    instance: wasmer::Instance,
    store: Store,
    wasi_env: wasmer_wasix::WasiFunctionEnv,
    memory: wasmer::Memory,
}

fn new_guest() -> Guest {
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

/// Calls `do_request_adapter_start`, returning the new subtask handle.
fn request_adapter_start(guest: &mut Guest, delay_ms: i32) -> u32 {
    let f = guest
        .instance
        .exports
        .get_function("do_request_adapter_start")
        .unwrap()
        .clone();
    let rc = f
        .call(
            &mut guest.store,
            &[Value::I32(SUBTASK_OUT_PTR), Value::I32(delay_ms)],
        )
        .expect("call do_request_adapter_start");
    assert_eq!(rc[0], Value::I32(0), "request_adapter_start should succeed");
    read_u32(&guest.memory, &guest.store, SUBTASK_OUT_PTR)
}

fn waitable_set_new(guest: &mut Guest) -> u32 {
    let f = guest
        .instance
        .exports
        .get_function("do_waitable_set_new")
        .unwrap()
        .clone();
    let rc = f
        .call(&mut guest.store, &[Value::I32(SET_OUT_PTR)])
        .expect("call do_waitable_set_new");
    assert_eq!(rc[0], Value::I32(0), "waitable_set_new should succeed");
    read_u32(&guest.memory, &guest.store, SET_OUT_PTR)
}

fn waitable_join(guest: &mut Guest, waitable: u32, set: u32) -> i32 {
    let f = guest
        .instance
        .exports
        .get_function("do_waitable_join")
        .unwrap()
        .clone();
    let rc = f
        .call(
            &mut guest.store,
            &[Value::I32(waitable as i32), Value::I32(set as i32)],
        )
        .expect("call do_waitable_join");
    let Value::I32(rc) = rc[0] else {
        panic!("do_waitable_join should return an i32");
    };
    rc
}

#[test]
fn waitable_set_wait_suspends_and_resumes_through_a_real_guest() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let mut guest = new_guest();

    let subtask = request_adapter_start(&mut guest, 0 /* default delay */);
    let set = waitable_set_new(&mut guest);
    assert_eq!(waitable_join(&mut guest, subtask, set), 0);

    let do_waitable_set_wait = guest
        .instance
        .exports
        .get_function("do_waitable_set_wait")
        .unwrap()
        .clone();

    // waitable_set_wait is the one real async host import: switch to the
    // async calling convention to drive it.
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

    // Proves the guest's coroutine genuinely suspended rather than the call
    // returning synchronously: real wall-clock time elapsed, matching the
    // simulated delay -- exactly Phase 0's verification technique. A small
    // tolerance absorbs scheduler/timer jitter under load (observed: a sleep
    // of exactly `DEFAULT_SIMULATED_DELAY` occasionally reports a fraction of
    // a millisecond short on a busy machine) without weakening what this
    // proves -- a synchronous, non-suspending call would return in
    // microseconds, not within a few ms of the full delay.
    const JITTER_TOLERANCE: Duration = Duration::from_millis(5);
    assert!(
        elapsed + JITTER_TOLERANCE >= DEFAULT_SIMULATED_DELAY,
        "expected waitable_set_wait to block for at least {DEFAULT_SIMULATED_DELAY:?} while the \
         guest was suspended; only took {elapsed:?}"
    );

    let resolved_subtask = read_u32(&guest.memory, &store, EVENT_OUT_PTR);
    let payload = read_u32(&guest.memory, &store, EVENT_OUT_PTR + 4);
    assert_eq!(
        resolved_subtask, subtask,
        "the event record's first u32 should be the resolved subtask's own handle"
    );
    assert_eq!(
        payload, FAKE_ADAPTER_ID,
        "resolved event payload should be the fake adapter id"
    );

    let do_subtask_drop = guest
        .instance
        .exports
        .get_function("do_subtask_drop")
        .unwrap()
        .clone();
    let rc = do_subtask_drop
        .call(&mut store, &[Value::I32(subtask as i32)])
        .expect("call do_subtask_drop");
    assert_eq!(rc[0], Value::I32(0), "subtask_drop should succeed");

    guest.wasi_env.on_exit(&mut store, None);
}

#[test]
fn waitable_join_rejects_once_the_set_is_at_capacity() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let mut guest = new_guest();
    let set = waitable_set_new(&mut guest);

    // Fill the set up to its cap -- every one of these joins must succeed.
    for i in 0..MAX_WAITABLES_PER_SET {
        let subtask = request_adapter_start(&mut guest, 0);
        assert_eq!(
            waitable_join(&mut guest, subtask, set),
            0,
            "join #{i} (within capacity) should succeed"
        );
    }

    // One more, past the cap, must be rejected -- not silently accepted,
    // not a panic.
    let one_too_many = request_adapter_start(&mut guest, 0);
    assert_eq!(
        waitable_join(&mut guest, one_too_many, set),
        -5,
        "join past MAX_WAITABLES_PER_SET should return SetOccupied (-5), not succeed"
    );

    let mut store = guest.store;
    guest.wasi_env.on_exit(&mut store, None);
}

#[test]
fn waitable_set_wait_returns_the_fastest_resolving_subtask_first() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let mut guest = new_guest();
    let set = waitable_set_new(&mut guest);

    // Three subtasks, started in this order, with deliberately
    // *non-monotonic* delays relative to start order -- if the
    // implementation were silently biased toward join order or handle
    // number instead of genuinely waiting for the fastest, this would catch
    // it: subtask "a" is started first but finishes last.
    let delays_ms = [(150, "a"), (50, "b"), (100, "c")];
    let mut subtasks = Vec::new();
    for (delay, _label) in delays_ms {
        let subtask = request_adapter_start(&mut guest, delay);
        assert_eq!(waitable_join(&mut guest, subtask, set), 0);
        subtasks.push(subtask);
    }
    let (subtask_a, subtask_b, subtask_c) = (subtasks[0], subtasks[1], subtasks[2]);

    let do_waitable_set_wait = guest
        .instance
        .exports
        .get_function("do_waitable_set_wait")
        .unwrap()
        .clone();

    let started = Instant::now();
    let mut resolution_order = Vec::new();
    let mut store = guest.store;
    for _ in 0..3 {
        let store_async = store.into_async();
        let result = runtime.block_on(do_waitable_set_wait.call_async(
            &store_async,
            vec![Value::I32(set as i32), Value::I32(EVENT_OUT_PTR)],
        ));
        store = store_async.into_store().expect(
            "store_async should be uniquely owned and unlocked once the call has completed",
        );
        let rc = result.expect("call do_waitable_set_wait");
        assert_eq!(rc[0], Value::I32(0), "waitable_set_wait should succeed");
        resolution_order.push(read_u32(&guest.memory, &store, EVENT_OUT_PTR));
    }
    let elapsed = started.elapsed();

    assert_eq!(
        resolution_order,
        vec![subtask_b, subtask_c, subtask_a],
        "waitable_set_wait should return the 50ms subtask (b) first, then the 100ms one (c), \
         then the 150ms one (a) -- resolution order, not start/join order"
    );

    // The three fake operations run concurrently in the background (each
    // spawned independently by request_adapter_start), so collecting all
    // three resolutions should take roughly as long as the *longest* delay
    // (150ms), not their sum (300ms) -- proving genuine concurrency, not
    // serialized fake work.
    assert!(
        elapsed < Duration::from_millis(250),
        "collecting all three events took {elapsed:?}, suggesting the fake operations ran \
         serialized rather than concurrently (expected roughly 150ms, well under their 300ms sum)"
    );

    guest.wasi_env.on_exit(&mut store, None);
}

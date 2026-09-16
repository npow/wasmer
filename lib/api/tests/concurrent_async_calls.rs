#![cfg(all(feature = "experimental-async", not(target_arch = "wasm32")))]

//! Discriminator test for a real open question left by this session's
//! wasi:webgpu Phase 2 scoping research: can two independent top-level
//! `Function::call_async` calls into the *same* `Store` ever be genuinely
//! concurrent (not merely both-eventually-succeed-but-serialized), or does
//! the store write lock / thread-local store-install guard in
//! `lib/api/src/backend/sys/async_runtime.rs` force full serialization --
//! or worse, panic (the `install_store_context` recursive-call guard is
//! thread-local, `CURRENT_CONTEXT`/`StoreContext`'s "current" tracking at
//! `async_runtime.rs:313` and `entities/store/context.rs:123`)?
//!
//! This is NOT the same question as "recursive `call_async` from inside an
//! import" (that's the documented, confirmed-panicking case). This is two
//! *sibling* top-level calls, e.g. two calls the embedder makes concurrently
//! from its own async code -- nothing in the existing test suite
//! (`jspi_async.rs`) exercises this; every existing test drives calls one at
//! a time.

use std::time::{Duration, Instant};

use anyhow::Result;
use wasmer::{Function, FunctionType, Instance, Module, Store, Type, Value, imports};

fn slow_echo_module() -> Vec<u8> {
    wat::parse_str(
        r#"
        (module
          (import "host" "slow" (func $slow (param i32) (result i32)))
          (func (export "entry") (param i32) (result i32)
            local.get 0
            call $slow))
        "#,
    )
    .expect("valid WAT module")
}

/// Two independent top-level `call_async` calls into the SAME store,
/// concurrent via `futures::future::join`, each suspending on a real host
/// future for `SLEEP` before returning. If the store forces full
/// serialization (or worse, panics), total elapsed will be close to
/// `2 * SLEEP`. If they genuinely interleave, elapsed will be close to
/// `1 * SLEEP`.
#[test]
#[cfg_attr(
    feature = "v8-default",
    ignore = "async functions are not supported by the default v8 backend"
)]
fn two_sibling_call_async_futures_on_one_store() -> Result<()> {
    const SLEEP: Duration = Duration::from_millis(150);

    let wasm = slow_echo_module();
    let mut store = Store::default();
    let module = Module::new(&store, wasm)?;

    let slow = Function::new_async(
        &mut store,
        FunctionType::new(vec![Type::I32], vec![Type::I32]),
        |values| {
            let values = values.to_vec();
            async move {
                tokio::time::sleep(SLEEP).await;
                Ok(values)
            }
        },
    );

    let import_object = imports! { "host" => { "slow" => slow } };
    let instance = Instance::new(&mut store, &module, &import_object)?;
    let entry = instance.exports.get_function("entry")?.clone();

    let store_async = store.into_async();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let start = Instant::now();
    let (r1, r2) = runtime.block_on(async {
        futures::future::join(
            entry.call_async(&store_async, vec![Value::I32(1)]),
            entry.call_async(&store_async, vec![Value::I32(2)]),
        )
        .await
    });
    let elapsed = start.elapsed();

    let r1 = r1.expect("first call_async should succeed, not panic or error");
    let r2 = r2.expect("second call_async should succeed, not panic or error");

    assert_eq!(r1[0], Value::I32(1));
    assert_eq!(r2[0], Value::I32(2));

    eprintln!(
        "two_sibling_call_async_futures_on_one_store: elapsed = {elapsed:?} (1x SLEEP = {SLEEP:?}, 2x SLEEP = {:?})",
        SLEEP * 2
    );

    // Real assertion, not just "did it panic": if the two calls were forced
    // fully serial, elapsed would be close to 2x SLEEP. Assert we're well
    // under that -- closer to 1x SLEEP -- as positive proof of genuine
    // concurrency, not just "didn't crash".
    assert!(
        elapsed < SLEEP + SLEEP / 2,
        "expected genuine concurrent interleaving (~{SLEEP:?}), got {elapsed:?} -- \
         looks like the store forced full serialization of the two calls"
    );

    Ok(())
}

/// Same probe, but on a single-threaded runtime -- rules out "it only works
/// because the two futures happened to run on different OS threads and the
/// thread-local store-install guard never saw the collision". If this also
/// shows real concurrency, the thread-local guard's cooperative-scheduling
/// argument (only one coroutine resume can be mid-flight on a given thread
/// at an instant, so sibling futures never actually collide even when
/// pinned to one thread) holds up in practice, not just in theory.
#[test]
#[cfg_attr(
    feature = "v8-default",
    ignore = "async functions are not supported by the default v8 backend"
)]
fn two_sibling_call_async_futures_on_one_store_single_threaded() -> Result<()> {
    const SLEEP: Duration = Duration::from_millis(150);

    let wasm = slow_echo_module();
    let mut store = Store::default();
    let module = Module::new(&store, wasm)?;

    let slow = Function::new_async(
        &mut store,
        FunctionType::new(vec![Type::I32], vec![Type::I32]),
        |values| {
            let values = values.to_vec();
            async move {
                tokio::time::sleep(SLEEP).await;
                Ok(values)
            }
        },
    );

    let import_object = imports! { "host" => { "slow" => slow } };
    let instance = Instance::new(&mut store, &module, &import_object)?;
    let entry = instance.exports.get_function("entry")?.clone();

    let store_async = store.into_async();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let start = Instant::now();
    let (r1, r2) = runtime.block_on(async {
        futures::future::join(
            entry.call_async(&store_async, vec![Value::I32(1)]),
            entry.call_async(&store_async, vec![Value::I32(2)]),
        )
        .await
    });
    let elapsed = start.elapsed();

    let r1 = r1.expect("first call_async should succeed on single-threaded runtime too");
    let r2 = r2.expect("second call_async should succeed on single-threaded runtime too");

    assert_eq!(r1[0], Value::I32(1));
    assert_eq!(r2[0], Value::I32(2));

    eprintln!("two_sibling_call_async_futures_on_one_store_single_threaded: elapsed = {elapsed:?}");

    assert!(
        elapsed < SLEEP + SLEEP / 2,
        "expected genuine concurrent interleaving even single-threaded (~{SLEEP:?}), got {elapsed:?}"
    );

    Ok(())
}

#![cfg(feature = "wasi-nn")]

//! End-to-end test for the `wasi_ephemeral_nn` host-function namespace: drives
//! a real guest module through `load` -> `init_execution_context` ->
//! `set_input` -> `compute` -> `get_output` as actual wasm imports, backed by
//! the real (non-mocked) `candle`-backed `CandleBackend`, on **both** `Cpu`
//! and (when the `wasi-nn-cuda` feature is on) `Gpu` execution targets. Unlike
//! `lib/wasi-nn/src/candle_backend.rs`'s unit tests -- which call the
//! `NnBackend` trait directly and never touch a wasm instance -- this exercises
//! the actual wasm-level ABI: raw i32 signatures and the
//! `WasmGraphBuilder`/`WasmTensor` guest-memory record layouts.
//!
//! The guest is a hand-written WAT module rather than a `wasixcc`/`cargo-wasix`
//! fixture (see `lib/wasix/tests/wasm_tests.rs`) -- neither tool is installed
//! on this machine. It declares no WASI imports beyond `wasi_ephemeral_nn` and
//! has no `_start`; it exports one thin wrapper function per import (each just
//! forwards its params to the import and returns the result), so the Rust
//! harness can drive the five-call sequence itself via `TypedFunction`,
//! writing input records into guest memory between calls. This sidesteps
//! doing multi-step control flow in WAT text, which is error-prone.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{Device, Tensor as CTensor};
use wasmer::{Module, Store, TypedFunction};
use wasmer_wasi_nn::candle_backend::CandleBackend;
use wasmer_wasi_nn::{ExecutionTarget, GraphEncoding, NnErrno, TensorType};
use wasmer_wasix::WasiEnv;

const WASI_EPHEMERAL_NN_WAT: &str = r#"
(module
  (import "wasi_ephemeral_nn" "load"
    (func $load (param i32 i32 i32 i32 i32) (result i32)))
  (import "wasi_ephemeral_nn" "load_by_name"
    (func $load_by_name (param i32 i32 i32) (result i32)))
  (import "wasi_ephemeral_nn" "init_execution_context"
    (func $init_execution_context (param i32 i32) (result i32)))
  (import "wasi_ephemeral_nn" "set_input"
    (func $set_input (param i32 i32 i32) (result i32)))
  (import "wasi_ephemeral_nn" "compute"
    (func $compute (param i32) (result i32)))
  (import "wasi_ephemeral_nn" "get_output"
    (func $get_output (param i32 i32 i32 i32 i32) (result i32)))

  (memory (export "memory") 2)

  (func (export "do_load")
        (param $builder_ptr i32) (param $builder_len i32)
        (param $encoding i32) (param $target i32) (param $graph_out i32)
        (result i32)
    local.get $builder_ptr
    local.get $builder_len
    local.get $encoding
    local.get $target
    local.get $graph_out
    call $load)

  (func (export "do_load_by_name")
        (param $name_ptr i32) (param $name_len i32) (param $graph_out i32)
        (result i32)
    local.get $name_ptr
    local.get $name_len
    local.get $graph_out
    call $load_by_name)

  (func (export "do_init_execution_context")
        (param $graph i32) (param $context_out i32)
        (result i32)
    local.get $graph
    local.get $context_out
    call $init_execution_context)

  (func (export "do_set_input")
        (param $context i32) (param $index i32) (param $tensor_ptr i32)
        (result i32)
    local.get $context
    local.get $index
    local.get $tensor_ptr
    call $set_input)

  (func (export "do_compute")
        (param $context i32)
        (result i32)
    local.get $context
    call $compute)

  (func (export "do_get_output")
        (param $context i32) (param $index i32) (param $out_buffer i32)
        (param $out_buffer_max_size i32) (param $bytes_written_out i32)
        (result i32)
    local.get $context
    local.get $index
    local.get $out_buffer
    local.get $out_buffer_max_size
    local.get $bytes_written_out
    call $get_output)
)
"#;

// Guest memory layout the Rust harness writes into (and reads results back
// from) between calls into the guest's thin `wasi_ephemeral_nn` wrappers.
// Generously spaced out; nothing here needs more than the module's two wasm
// pages (128 KiB). `load`'s `$graph_builder_array` now carries two blobs
// (config JSON, then safetensors weights -- see `CandleBackend`'s doc
// comment), so it needs two `WasmGraphBuilder` records back to back.
const GRAPH_BUILDER_ARRAY_OFFSET: u32 = 0; // 2x WasmGraphBuilder { ptr: u32, len: u32 }, 16 bytes
const GRAPH_OUT_OFFSET: u32 = 16; // u32 out-param, 4 bytes
const CONTEXT_OUT_OFFSET: u32 = 20; // u32 out-param, 4 bytes
const DIMS_OFFSET: u32 = 24; // one u32 dimension value, 4 bytes
const NAME_OFFSET: u32 = 32; // `load_by_name`'s name bytes
const INPUT_DATA_OFFSET: u32 = 64; // two f32, 8 bytes
const TENSOR_OFFSET: u32 = 96; // WasmTensor record, 20 bytes
const OUTPUT_BUFFER_OFFSET: u32 = 128; // room for output (3 f32 = 12 bytes needed)
const OUTPUT_BUFFER_CAPACITY: i32 = 64;
const BYTES_WRITTEN_OFFSET: u32 = 256; // u32 out-param, 4 bytes
const CONFIG_BYTES_OFFSET: u32 = 4096; // config.json bytes
const WEIGHTS_BYTES_OFFSET: u32 = 8192; // safetensors blob, clear of the config

/// Packs a `WasmGraphBuilder { ptr: u32, len: u32 }` record: `#[repr(C)]`,
/// natural alignment, 8 bytes total. That type is private to
/// `wasmer_wasix::syscalls::wasi_nn::types`, so its layout is re-derived here
/// byte-for-byte (verified against that file) rather than imported.
fn pack_graph_builder(ptr: u32, len: u32) -> [u8; 8] {
    let mut buf = [0u8; 8];
    buf[0..4].copy_from_slice(&ptr.to_le_bytes());
    buf[4..8].copy_from_slice(&len.to_le_bytes());
    buf
}

/// Packs a `WasmTensor { dimensions_ptr: u32, dimensions_len: u32, ty: u8,
/// data_ptr: u32, data_len: u32 }` record: `#[repr(C)]` natural alignment
/// pads 3 bytes after `ty` so `data_ptr` lands 4-byte aligned at offset 12;
/// 20 bytes total.
fn pack_tensor(
    dimensions_ptr: u32,
    dimensions_len: u32,
    ty: u8,
    data_ptr: u32,
    data_len: u32,
) -> [u8; 20] {
    let mut buf = [0u8; 20];
    buf[0..4].copy_from_slice(&dimensions_ptr.to_le_bytes());
    buf[4..8].copy_from_slice(&dimensions_len.to_le_bytes());
    buf[8] = ty;
    // Bytes 9..12 are `#[repr(C)]` padding; left zeroed.
    buf[12..16].copy_from_slice(&data_ptr.to_le_bytes());
    buf[16..20].copy_from_slice(&data_len.to_le_bytes());
    buf
}

/// Builds a tiny `y = x @ W^T + b` (config_json, safetensors_bytes) pair --
/// a one-layer instance of `CandleBackend`'s general layer-stack format, with
/// no network access and no external model file. Same shape as
/// `candle_backend::tests::tiny_mlp`, minus the second layer, since this test
/// only needs to prove the wasm-level ABI plumbing, not model generality
/// (that's `candle_backend`'s job -- this just has to load *some* valid
/// two-blob graph).
fn tiny_linear_graph() -> (Vec<u8>, Vec<u8>) {
    let dev = Device::Cpu;
    // W: [[1, 0], [0, 1], [1, 1]] (3x2), b: [0, 0, 1]
    let weight = CTensor::from_vec(vec![1f32, 0., 0., 1., 1., 1.], (3, 2), &dev).unwrap();
    let bias = CTensor::from_vec(vec![0f32, 0., 1.], 3, &dev).unwrap();
    let mut tensors = HashMap::new();
    tensors.insert("weight".to_string(), weight);
    tensors.insert("bias".to_string(), bias);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    candle_core::safetensors::save(&tensors, &path).unwrap();
    let weights_bytes = std::fs::read(path).unwrap();

    let config = serde_json::json!({
        "layers": [{"type": "linear", "weight": "weight", "bias": "bias"}]
    });
    let config_bytes = serde_json::to_vec(&config).unwrap();
    (config_bytes, weights_bytes)
}

/// Runs the full `load` -> `init_execution_context` -> `set_input` ->
/// `compute` -> `get_output` sequence against a real wasm guest and the real
/// `CandleBackend`, on whichever `target` is passed in, and returns the
/// decoded output tensor.
fn run_wasi_ephemeral_nn(target: ExecutionTarget) -> Vec<f32> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let mut store = Store::default();
    let module = Module::new(&store, WASI_EPHEMERAL_NN_WAT).expect("compile guest WAT module");

    let mut builder = WasiEnv::builder("wasi-nn-test").engine(store.engine().clone());
    builder.capabilities_mut().nn.allow = true;
    builder.capabilities_mut().nn.allow_gpu = true;
    builder.set_nn_backend(Arc::new(CandleBackend::new()));

    let (instance, wasi_env) = builder
        .instantiate(module, &mut store)
        .expect("instantiate guest module with wasi_ephemeral_nn imports");

    let memory = instance
        .exports
        .get_memory("memory")
        .expect("guest exports memory")
        .clone();

    let do_load: TypedFunction<(i32, i32, i32, i32, i32), i32> = instance
        .exports
        .get_typed_function(&store, "do_load")
        .expect("do_load export");
    let do_load_by_name: TypedFunction<(i32, i32, i32), i32> = instance
        .exports
        .get_typed_function(&store, "do_load_by_name")
        .expect("do_load_by_name export");
    let do_init_execution_context: TypedFunction<(i32, i32), i32> = instance
        .exports
        .get_typed_function(&store, "do_init_execution_context")
        .expect("do_init_execution_context export");
    let do_set_input: TypedFunction<(i32, i32, i32), i32> = instance
        .exports
        .get_typed_function(&store, "do_set_input")
        .expect("do_set_input export");
    let do_compute: TypedFunction<i32, i32> = instance
        .exports
        .get_typed_function(&store, "do_compute")
        .expect("do_compute export");
    let do_get_output: TypedFunction<(i32, i32, i32, i32, i32), i32> = instance
        .exports
        .get_typed_function(&store, "do_get_output")
        .expect("do_get_output export");

    // -- load() --
    let (config_bytes, weights_bytes) = tiny_linear_graph();
    {
        let view = memory.view(&store);
        view.write(CONFIG_BYTES_OFFSET as u64, &config_bytes)
            .expect("write config bytes");
        view.write(WEIGHTS_BYTES_OFFSET as u64, &weights_bytes)
            .expect("write weights bytes");
        let builder_records = [
            pack_graph_builder(CONFIG_BYTES_OFFSET, config_bytes.len() as u32),
            pack_graph_builder(WEIGHTS_BYTES_OFFSET, weights_bytes.len() as u32),
        ]
        .concat();
        view.write(GRAPH_BUILDER_ARRAY_OFFSET as u64, &builder_records)
            .expect("write graph builder array");
    }

    let errno = do_load
        .call(
            &mut store,
            GRAPH_BUILDER_ARRAY_OFFSET as i32,
            2, // two graph_builder blobs: config, then weights
            GraphEncoding::Autodetect as i32,
            target as i32,
            GRAPH_OUT_OFFSET as i32,
        )
        .expect("call do_load") as u32;
    assert_eq!(
        errno,
        NnErrno::Success.to_u32(),
        "load failed with errno {errno} (target {target:?})"
    );

    let graph = {
        let view = memory.view(&store);
        let mut graph_bytes = [0u8; 4];
        view.read(GRAPH_OUT_OFFSET as u64, &mut graph_bytes)
            .expect("read graph handle");
        u32::from_le_bytes(graph_bytes)
    };

    // -- init_execution_context() --
    let errno = do_init_execution_context
        .call(&mut store, graph as i32, CONTEXT_OUT_OFFSET as i32)
        .expect("call do_init_execution_context") as u32;
    assert_eq!(
        errno,
        NnErrno::Success.to_u32(),
        "init_execution_context failed with errno {errno}"
    );

    let context = {
        let view = memory.view(&store);
        let mut context_bytes = [0u8; 4];
        view.read(CONTEXT_OUT_OFFSET as u64, &mut context_bytes)
            .expect("read execution context handle");
        u32::from_le_bytes(context_bytes)
    };

    // -- set_input() --
    {
        let view = memory.view(&store);
        view.write(DIMS_OFFSET as u64, &2u32.to_le_bytes())
            .expect("write tensor dimensions");
        let input: [f32; 2] = [3.0, 4.0];
        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
        view.write(INPUT_DATA_OFFSET as u64, &input_bytes)
            .expect("write tensor data");
        let tensor_record = pack_tensor(
            DIMS_OFFSET,
            1, // one dimension value (the vector's length, 2)
            TensorType::F32 as u8,
            INPUT_DATA_OFFSET,
            input_bytes.len() as u32,
        );
        view.write(TENSOR_OFFSET as u64, &tensor_record)
            .expect("write tensor record");
    }

    let errno = do_set_input
        .call(&mut store, context as i32, 0, TENSOR_OFFSET as i32)
        .expect("call do_set_input") as u32;
    assert_eq!(
        errno,
        NnErrno::Success.to_u32(),
        "set_input failed with errno {errno}"
    );

    // -- compute() --
    let errno = do_compute
        .call(&mut store, context as i32)
        .expect("call do_compute") as u32;
    assert_eq!(
        errno,
        NnErrno::Success.to_u32(),
        "compute failed with errno {errno}"
    );

    // -- get_output() --
    let errno = do_get_output
        .call(
            &mut store,
            context as i32,
            0,
            OUTPUT_BUFFER_OFFSET as i32,
            OUTPUT_BUFFER_CAPACITY,
            BYTES_WRITTEN_OFFSET as i32,
        )
        .expect("call do_get_output") as u32;
    assert_eq!(
        errno,
        NnErrno::Success.to_u32(),
        "get_output failed with errno {errno}"
    );

    let output: Vec<f32> = {
        let view = memory.view(&store);
        let mut written_bytes = [0u8; 4];
        view.read(BYTES_WRITTEN_OFFSET as u64, &mut written_bytes)
            .expect("read bytes_written_out");
        let written = u32::from_le_bytes(written_bytes);
        assert_eq!(written, 12, "expected 3 f32 output elements (12 bytes)");

        let mut output_bytes = vec![0u8; written as usize];
        view.read(OUTPUT_BUFFER_OFFSET as u64, &mut output_bytes)
            .expect("read output bytes");
        output_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    // -- load_by_name(), bonus coverage of the sixth import: `CandleBackend`
    // doesn't implement it, so the default `NnBackend::load_by_name` must
    // reject with `not_found`.
    {
        let view = memory.view(&store);
        view.write(NAME_OFFSET as u64, b"nonexistent-graph")
            .expect("write graph name");
    }
    let errno = do_load_by_name
        .call(
            &mut store,
            NAME_OFFSET as i32,
            "nonexistent-graph".len() as i32,
            GRAPH_OUT_OFFSET as i32,
        )
        .expect("call do_load_by_name") as u32;
    assert_eq!(
        errno,
        NnErrno::NotFound.to_u32(),
        "load_by_name should reject with not_found, got errno {errno}"
    );

    wasi_env.on_exit(&mut store, None);
    output
}

#[test]
fn wasi_ephemeral_nn_end_to_end_cpu_inference() {
    let output = run_wasi_ephemeral_nn(ExecutionTarget::Cpu);
    // [3,4] @ [[1,0],[0,1],[1,1]]^T + [0,0,1] = [3, 4, 3+4+1] = [3, 4, 8]
    assert_eq!(output, vec![3.0, 4.0, 8.0]);
}

/// Closes the CPU-only gap in the original version of this test: the same
/// wasm-level `wasi_ephemeral_nn` sequence, but with `target = gpu`, so this
/// actually proves a `.wasm` guest can drive real CUDA compute through the
/// host bridge -- not just `CandleBackend`'s own Rust-level unit test.
#[cfg(feature = "wasi-nn-cuda")]
#[test]
fn wasi_ephemeral_nn_end_to_end_gpu_inference() {
    let output = run_wasi_ephemeral_nn(ExecutionTarget::Gpu);
    assert_eq!(output, vec![3.0, 4.0, 8.0]);
}

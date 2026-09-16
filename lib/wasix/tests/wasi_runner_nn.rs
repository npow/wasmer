#![cfg(feature = "wasi-nn")]

//! Proves the `WasiRunner` <-> wasi-nn hook: `WasiRunner::with_nn_backend`
//! plus `capabilities_mut().nn` actually reach a guest instantiated through
//! `WasiRunner::prepare_webc_env` (the webc/package run path), not just the
//! raw-`.wasm` CLI path (`lib/cli/src/commands/run/wasi.rs`'s
//! `Wasi::prepare`, covered by `tests/wasi_nn.rs`).
//!
//! Drives the same `load` -> `init_execution_context` -> `set_input` ->
//! `compute` -> `get_output` sequence as `tests/wasi_nn.rs`, against the same
//! kind of hand-written WAT guest and tiny one-layer `CandleBackend` graph,
//! but obtains the `WasiEnvBuilder` from `WasiRunner::prepare_webc_env`
//! instead of `WasiEnv::builder()` directly.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{Device, Tensor as CTensor};
use wasmer::{Module, Store, TypedFunction};
use wasmer_types::ModuleHash;
use wasmer_wasi_nn::candle_backend::CandleBackend;
use wasmer_wasi_nn::{ExecutionTarget, GraphEncoding, NnErrno, TensorType};
use wasmer_wasix::runners::wasi::{PackageOrHash, RuntimeOrEngine, WasiRunner};
use webc::metadata::annotations::Wasi as WasiAnnotations;

const WASI_EPHEMERAL_NN_WAT: &str = r#"
(module
  (import "wasi_ephemeral_nn" "load"
    (func $load (param i32 i32 i32 i32 i32) (result i32)))
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

const GRAPH_BUILDER_ARRAY_OFFSET: u32 = 0; // 2x WasmGraphBuilder { ptr: u32, len: u32 }, 16 bytes
const GRAPH_OUT_OFFSET: u32 = 16;
const CONTEXT_OUT_OFFSET: u32 = 20;
const DIMS_OFFSET: u32 = 24;
const INPUT_DATA_OFFSET: u32 = 64;
const TENSOR_OFFSET: u32 = 96;
const OUTPUT_BUFFER_OFFSET: u32 = 128;
const OUTPUT_BUFFER_CAPACITY: i32 = 64;
const BYTES_WRITTEN_OFFSET: u32 = 256;
const CONFIG_BYTES_OFFSET: u32 = 4096;
const WEIGHTS_BYTES_OFFSET: u32 = 8192;

/// See `tests/wasi_nn.rs::pack_graph_builder` -- identical layout, duplicated
/// here since integration test binaries don't share code.
fn pack_graph_builder(ptr: u32, len: u32) -> [u8; 8] {
    let mut buf = [0u8; 8];
    buf[0..4].copy_from_slice(&ptr.to_le_bytes());
    buf[4..8].copy_from_slice(&len.to_le_bytes());
    buf
}

/// See `tests/wasi_nn.rs::pack_tensor` -- identical layout.
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
    buf[12..16].copy_from_slice(&data_ptr.to_le_bytes());
    buf[16..20].copy_from_slice(&data_len.to_le_bytes());
    buf
}

/// See `tests/wasi_nn.rs::tiny_linear_graph` -- identical one-layer graph.
fn tiny_linear_graph() -> (Vec<u8>, Vec<u8>) {
    let dev = Device::Cpu;
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

#[test]
fn wasi_runner_wires_nn_backend_into_webc_env() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();

    let mut store = Store::default();
    let module = Module::new(&store, WASI_EPHEMERAL_NN_WAT).expect("compile guest WAT module");

    // The hook under test: attach a real backend and enable the capability
    // through `WasiRunner`, then obtain a `WasiEnvBuilder` the same way the
    // webc/package run path does (`prepare_webc_env`), not `WasiEnv::builder()`.
    let mut runner = WasiRunner::new();
    runner.capabilities_mut().nn.allow = true;
    runner.capabilities_mut().nn.allow_gpu = false;
    runner.with_nn_backend(Arc::new(CandleBackend::new()));

    let wasi_annotations = WasiAnnotations::new("wasi-runner-nn-test");
    let builder = runner
        .prepare_webc_env(
            "wasi-runner-nn-test",
            &wasi_annotations,
            PackageOrHash::Hash(ModuleHash::random()),
            RuntimeOrEngine::Engine(store.engine().clone()),
            None,
        )
        .expect("prepare_webc_env should succeed");

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
            2,
            GraphEncoding::Autodetect as i32,
            ExecutionTarget::Cpu as i32,
            GRAPH_OUT_OFFSET as i32,
        )
        .expect("call do_load") as u32;
    assert_eq!(
        errno,
        NnErrno::Success.to_u32(),
        "load failed with errno {errno} -- the nn_backend hook did not reach the WasiRunner-built env"
    );

    let graph = {
        let view = memory.view(&store);
        let mut graph_bytes = [0u8; 4];
        view.read(GRAPH_OUT_OFFSET as u64, &mut graph_bytes)
            .expect("read graph handle");
        u32::from_le_bytes(graph_bytes)
    };

    let errno = do_init_execution_context
        .call(&mut store, graph as i32, CONTEXT_OUT_OFFSET as i32)
        .expect("call do_init_execution_context") as u32;
    assert_eq!(errno, NnErrno::Success.to_u32());

    let context = {
        let view = memory.view(&store);
        let mut context_bytes = [0u8; 4];
        view.read(CONTEXT_OUT_OFFSET as u64, &mut context_bytes)
            .expect("read execution context handle");
        u32::from_le_bytes(context_bytes)
    };

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
            1,
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
    assert_eq!(errno, NnErrno::Success.to_u32());

    let errno = do_compute
        .call(&mut store, context as i32)
        .expect("call do_compute") as u32;
    assert_eq!(errno, NnErrno::Success.to_u32());

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
    assert_eq!(errno, NnErrno::Success.to_u32());

    let output: Vec<f32> = {
        let view = memory.view(&store);
        let mut written_bytes = [0u8; 4];
        view.read(BYTES_WRITTEN_OFFSET as u64, &mut written_bytes)
            .expect("read bytes_written_out");
        let written = u32::from_le_bytes(written_bytes);
        assert_eq!(written, 12);

        let mut output_bytes = vec![0u8; written as usize];
        view.read(OUTPUT_BUFFER_OFFSET as u64, &mut output_bytes)
            .expect("read output bytes");
        output_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    // [3,4] @ [[1,0],[0,1],[1,1]]^T + [0,0,1] = [3, 4, 3+4+1] = [3, 4, 8]
    assert_eq!(output, vec![3.0, 4.0, 8.0]);

    wasi_env.on_exit(&mut store, None);
}

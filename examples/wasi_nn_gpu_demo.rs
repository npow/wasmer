//! Runs a real, PyTorch-trained model through the `wasi_ephemeral_nn`
//! (wasi-nn) host-function bridge, executing inference on the GPU when this
//! binary was built with `wasi-nn-demo-cuda` (falls back to CPU otherwise).
//!
//! The model (`examples/assets/digits_mlp/`) is a small MLP handwriting-digit
//! classifier: `Linear(64,32) -> ReLU -> Linear(32,10)`, trained for real in
//! PyTorch on scikit-learn's digits dataset (96.7% held-out test accuracy --
//! see `model.safetensors`/`config.json`, and the training script this repo's
//! author ran to produce them). `samples.json` holds ten real held-out test
//! images plus their true labels.
//!
//! What's real here: the model, the trained weights, the GPU compute. What's
//! NOT wasm: the actual matrix math, which runs natively in `candle` (the
//! wasm guest is a thin client that calls `load`/`set_input`/`compute`/
//! `get_output` -- see `CandleBackend`'s doc comment for why wasi-nn works
//! this way, same as every real-world wasi-nn deployment).
//!
//! ```shell
//! cargo run --example wasi-nn-gpu-demo --release --features "cranelift,wasi-nn-demo-cuda"
//! ```

use std::sync::Arc;

use anyhow::{Context, Result};
use wasmer::{Module, Store, TypedFunction};
use wasmer_wasi_nn::candle_backend::CandleBackend;
use wasmer_wasi_nn::{ExecutionTarget, GraphEncoding, NnErrno, TensorType};
use wasmer_wasix::WasiEnv;

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

  (memory (export "memory") 4)

  (func (export "do_load")
        (param $builder_ptr i32) (param $builder_len i32)
        (param $encoding i32) (param $target i32) (param $graph_out i32)
        (result i32)
    local.get $builder_ptr local.get $builder_len
    local.get $encoding local.get $target local.get $graph_out
    call $load)

  (func (export "do_init_execution_context")
        (param $graph i32) (param $context_out i32) (result i32)
    local.get $graph local.get $context_out
    call $init_execution_context)

  (func (export "do_set_input")
        (param $context i32) (param $index i32) (param $tensor_ptr i32) (result i32)
    local.get $context local.get $index local.get $tensor_ptr
    call $set_input)

  (func (export "do_compute") (param $context i32) (result i32)
    local.get $context call $compute)

  (func (export "do_get_output")
        (param $context i32) (param $index i32) (param $out_buffer i32)
        (param $out_buffer_max_size i32) (param $bytes_written_out i32) (result i32)
    local.get $context local.get $index local.get $out_buffer
    local.get $out_buffer_max_size local.get $bytes_written_out
    call $get_output)
)
"#;

// Guest memory layout the host writes into (and reads results back from).
const GRAPH_BUILDER_ARRAY_OFFSET: u32 = 0; // 2x WasmGraphBuilder{ptr,len}, 16 bytes
const GRAPH_OUT_OFFSET: u32 = 16;
const CONTEXT_OUT_OFFSET: u32 = 20;
const DIMS_OFFSET: u32 = 24; // one u32 (64)
const TENSOR_OFFSET: u32 = 32; // WasmTensor record, 20 bytes
const INPUT_DATA_OFFSET: u32 = 64; // 64 f32 = 256 bytes
const OUTPUT_BUFFER_OFFSET: u32 = 1024; // 10 f32 logits = 40 bytes needed
const OUTPUT_BUFFER_CAPACITY: i32 = 256;
const BYTES_WRITTEN_OFFSET: u32 = 2048;
const CONFIG_BYTES_OFFSET: u32 = 4096;
const WEIGHTS_BYTES_OFFSET: u32 = 8192;

fn pack_graph_builder(ptr: u32, len: u32) -> [u8; 8] {
    let mut buf = [0u8; 8];
    buf[0..4].copy_from_slice(&ptr.to_le_bytes());
    buf[4..8].copy_from_slice(&len.to_le_bytes());
    buf
}

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

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap()
}

fn main() -> Result<()> {
    let assets =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/digits_mlp");
    let config_bytes = std::fs::read(assets.join("config.json")).context("read config.json")?;
    let weights_bytes =
        std::fs::read(assets.join("model.safetensors")).context("read model.safetensors")?;
    let samples: serde_json::Value = serde_json::from_slice(
        &std::fs::read(assets.join("samples.json")).context("read samples.json")?,
    )?;
    let inputs = samples["inputs"].as_array().context("samples.inputs")?;
    let labels = samples["labels"].as_array().context("samples.labels")?;

    let target = if cfg!(feature = "wasi-nn-demo-cuda") {
        ExecutionTarget::Gpu
    } else {
        ExecutionTarget::Cpu
    };
    println!("wasi-nn digit classifier demo -- target: {target:?}");
    println!(
        "model: Linear(64,32) -> ReLU -> Linear(32,10), trained in PyTorch (96.7% held-out test accuracy)\n"
    );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let _guard = runtime.enter();

    let mut store = Store::default();
    let module = Module::new(&store, WASI_EPHEMERAL_NN_WAT)?;

    let mut builder = WasiEnv::builder("wasi-nn-gpu-demo").engine(store.engine().clone());
    builder.capabilities_mut().nn.allow = true;
    builder.capabilities_mut().nn.allow_gpu = true;
    builder.set_nn_backend(Arc::new(CandleBackend::new()));
    let (instance, wasi_env) = builder.instantiate(module, &mut store)?;

    let memory = instance.exports.get_memory("memory")?.clone();
    let do_load: TypedFunction<(i32, i32, i32, i32, i32), i32> =
        instance.exports.get_typed_function(&store, "do_load")?;
    let do_init_execution_context: TypedFunction<(i32, i32), i32> = instance
        .exports
        .get_typed_function(&store, "do_init_execution_context")?;
    let do_set_input: TypedFunction<(i32, i32, i32), i32> = instance
        .exports
        .get_typed_function(&store, "do_set_input")?;
    let do_compute: TypedFunction<i32, i32> =
        instance.exports.get_typed_function(&store, "do_compute")?;
    let do_get_output: TypedFunction<(i32, i32, i32, i32, i32), i32> = instance
        .exports
        .get_typed_function(&store, "do_get_output")?;

    // -- load() the real trained model, once --
    {
        let view = memory.view(&store);
        view.write(CONFIG_BYTES_OFFSET as u64, &config_bytes)?;
        view.write(WEIGHTS_BYTES_OFFSET as u64, &weights_bytes)?;
        let records = [
            pack_graph_builder(CONFIG_BYTES_OFFSET, config_bytes.len() as u32),
            pack_graph_builder(WEIGHTS_BYTES_OFFSET, weights_bytes.len() as u32),
        ]
        .concat();
        view.write(GRAPH_BUILDER_ARRAY_OFFSET as u64, &records)?;
    }
    let errno = do_load.call(
        &mut store,
        GRAPH_BUILDER_ARRAY_OFFSET as i32,
        2,
        GraphEncoding::Autodetect as i32,
        target as i32,
        GRAPH_OUT_OFFSET as i32,
    )? as u32;
    anyhow::ensure!(
        errno == NnErrno::Success.to_u32(),
        "load failed with errno {errno}"
    );
    let graph = {
        let mut b = [0u8; 4];
        memory.view(&store).read(GRAPH_OUT_OFFSET as u64, &mut b)?;
        u32::from_le_bytes(b)
    };

    let errno =
        do_init_execution_context.call(&mut store, graph as i32, CONTEXT_OUT_OFFSET as i32)? as u32;
    anyhow::ensure!(
        errno == NnErrno::Success.to_u32(),
        "init_execution_context failed with errno {errno}"
    );
    let context = {
        let mut b = [0u8; 4];
        memory
            .view(&store)
            .read(CONTEXT_OUT_OFFSET as u64, &mut b)?;
        u32::from_le_bytes(b)
    };

    // -- run each held-out sample through the one loaded execution context --
    let mut correct = 0usize;
    for (i, (input, label)) in inputs.iter().zip(labels.iter()).enumerate() {
        let input: Vec<f32> = input
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let label = label.as_u64().unwrap() as usize;

        {
            let view = memory.view(&store);
            view.write(DIMS_OFFSET as u64, &(input.len() as u32).to_le_bytes())?;
            view.write(INPUT_DATA_OFFSET as u64, &f32_to_bytes(&input))?;
            let tensor = pack_tensor(
                DIMS_OFFSET,
                1,
                TensorType::F32 as u8,
                INPUT_DATA_OFFSET,
                (input.len() * 4) as u32,
            );
            view.write(TENSOR_OFFSET as u64, &tensor)?;
        }

        let errno = do_set_input.call(&mut store, context as i32, 0, TENSOR_OFFSET as i32)? as u32;
        anyhow::ensure!(
            errno == NnErrno::Success.to_u32(),
            "set_input failed with errno {errno}"
        );

        let errno = do_compute.call(&mut store, context as i32)? as u32;
        anyhow::ensure!(
            errno == NnErrno::Success.to_u32(),
            "compute failed with errno {errno}"
        );

        let errno = do_get_output.call(
            &mut store,
            context as i32,
            0,
            OUTPUT_BUFFER_OFFSET as i32,
            OUTPUT_BUFFER_CAPACITY,
            BYTES_WRITTEN_OFFSET as i32,
        )? as u32;
        anyhow::ensure!(
            errno == NnErrno::Success.to_u32(),
            "get_output failed with errno {errno}"
        );

        let logits: Vec<f32> = {
            let view = memory.view(&store);
            let mut written = [0u8; 4];
            view.read(BYTES_WRITTEN_OFFSET as u64, &mut written)?;
            let written = u32::from_le_bytes(written) as usize;
            let mut bytes = vec![0u8; written];
            view.read(OUTPUT_BUFFER_OFFSET as u64, &mut bytes)?;
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        let predicted = argmax(&logits);
        let mark = if predicted == label {
            correct += 1;
            "✓"
        } else {
            "✗"
        };
        println!("sample {i}: predicted={predicted} actual={label} {mark}");
    }

    println!(
        "\n{correct}/{} correct on these held-out samples, running on {target:?}",
        inputs.len()
    );

    wasi_env.on_exit(&mut store, None);
    Ok(())
}

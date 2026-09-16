//! Extends the `gpu-bridge` example's buffer/pipeline handle ABI (see
//! `gpu_bridge.rs`) to show a guest *owning a training loop*, not just one
//! dispatch: the wasm guest runs full-batch gradient descent for a
//! single-feature linear model (`y = w*x + b`) entirely on the GPU, via two
//! WGSL kernels it supplies itself — one that computes the forward pass and
//! gradients in a single dispatch (the model and batch are tiny, so a
//! single-invocation reduction avoids needing atomics/scan machinery), and
//! one that applies the gradient-descent update. The guest uploads its
//! training state once, then loops entirely on the GPU (no per-step
//! host round trip) before reading the final state back.
//!
//! The host still never sees the model: it only supplies the data (as raw
//! bytes at a memory offset) and reads back the trained parameters and loss.
//!
//! ```shell
//! cargo run --example gpu-bridge-train --release --features "cranelift,gpu-bridge-train-example"
//! ```

use anyhow::Result;
use std::collections::HashMap;
use wasmer::{
    AsStoreRef, Function, FunctionEnv, FunctionEnvMut, Instance, Memory, MemoryView, Module, Store,
    TypedFunction, imports, wat2wasm,
};
use wgpu::util::DeviceExt;

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Call succeeded.
const SUCCESS: i32 = 0;
/// A `(ptr, len)`/offset pair fell outside guest memory or a buffer's bounds.
const ERR_RANGE: i32 = -1;
/// The handle was never issued, or was already released.
const ERR_HANDLE: i32 = -2;

/// Per-instance GPU resource tables: guest-owned buffers and compute
/// pipelines, addressed by opaque handles the guest holds onto across calls.
/// One real `wgpu` device/queue is set up once and shared by every call.
///
/// Duplicated from `gpu_bridge.rs` rather than shared: each example file in
/// this directory is self-contained, and the two bridges are small enough
/// that a shared module would add more indirection than it saves.
struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: HashMap<u32, wgpu::Buffer>,
    pipelines: HashMap<u32, wgpu::ComputePipeline>,
    next_buffer: u32,
    next_pipeline: u32,
}

impl GpuState {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("no Vulkan adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("failed to create GPU device");

        Self {
            device,
            queue,
            buffers: HashMap::new(),
            pipelines: HashMap::new(),
            next_buffer: 0,
            next_pipeline: 0,
        }
    }

    fn alloc_buffer(&mut self, buffer: wgpu::Buffer) -> u32 {
        let handle = self.next_buffer;
        self.next_buffer += 1;
        self.buffers.insert(handle, buffer);
        handle
    }

    fn alloc_pipeline(&mut self, pipeline: wgpu::ComputePipeline) -> u32 {
        let handle = self.next_pipeline;
        self.next_pipeline += 1;
        self.pipelines.insert(handle, pipeline);
        handle
    }
}

/// Environment shared with every `env` import: the guest's exported memory,
/// plus the GPU resource tables.
pub struct GpuBridgeEnv {
    memory: Option<Memory>,
    gpu: GpuState,
}

impl GpuBridgeEnv {
    fn set_memory(&mut self, memory: Memory) {
        self.memory = Some(memory);
    }

    fn view<'a>(&'a self, store: &'a impl AsStoreRef) -> MemoryView<'a> {
        self.memory.as_ref().unwrap().view(store)
    }
}

fn buffer_upload(mut ctx: FunctionEnvMut<GpuBridgeEnv>, ptr: u32, len: u32) -> i32 {
    let mut bytes = vec![0u8; len as usize];
    if ctx.data().view(&ctx).read(ptr as u64, &mut bytes).is_err() {
        return ERR_RANGE;
    }

    let buffer = ctx
        .data()
        .gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("guest-buffer"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
    ctx.data_mut().gpu.alloc_buffer(buffer) as i32
}

fn buffer_read(ctx: FunctionEnvMut<GpuBridgeEnv>, handle: u32, ptr: u32, len: u32) -> i32 {
    let data = ctx.data();
    let Some(buffer) = data.gpu.buffers.get(&handle) else {
        return ERR_HANDLE;
    };
    let byte_len = len as wgpu::BufferAddress;
    if byte_len > buffer.size() {
        return ERR_RANGE;
    }

    let staging = data.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = data
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_len);
    data.gpu.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("map_async receiver dropped");
    });
    data.gpu
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
    rx.recv()
        .expect("map_async callback never ran")
        .expect("failed to map staging buffer");

    let bytes = slice.get_mapped_range().expect("no mapped range").to_vec();
    staging.unmap();

    if ctx.data().view(&ctx).write(ptr as u64, &bytes).is_err() {
        return ERR_RANGE;
    }
    SUCCESS
}

fn buffer_write(
    ctx: FunctionEnvMut<GpuBridgeEnv>,
    handle: u32,
    offset: u32,
    ptr: u32,
    len: u32,
) -> i32 {
    let mut bytes = vec![0u8; len as usize];
    if ctx.data().view(&ctx).read(ptr as u64, &mut bytes).is_err() {
        return ERR_RANGE;
    }

    let data = ctx.data();
    let Some(buffer) = data.gpu.buffers.get(&handle) else {
        return ERR_HANDLE;
    };
    if offset as wgpu::BufferAddress + bytes.len() as wgpu::BufferAddress > buffer.size() {
        return ERR_RANGE;
    }
    data.gpu
        .queue
        .write_buffer(buffer, offset as wgpu::BufferAddress, &bytes);
    SUCCESS
}

fn buffer_release(mut ctx: FunctionEnvMut<GpuBridgeEnv>, handle: u32) -> i32 {
    match ctx.data_mut().gpu.buffers.remove(&handle) {
        Some(_) => SUCCESS,
        None => ERR_HANDLE,
    }
}

fn pipeline_create(mut ctx: FunctionEnvMut<GpuBridgeEnv>, ptr: u32, len: u32) -> i32 {
    let mut bytes = vec![0u8; len as usize];
    if ctx.data().view(&ctx).read(ptr as u64, &mut bytes).is_err() {
        return ERR_RANGE;
    }
    let Ok(source) = String::from_utf8(bytes) else {
        return ERR_RANGE;
    };

    let shader = ctx
        .data()
        .gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("guest-shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline =
        ctx.data()
            .gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("guest-pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
    ctx.data_mut().gpu.alloc_pipeline(pipeline) as i32
}

fn pipeline_dispatch(
    ctx: FunctionEnvMut<GpuBridgeEnv>,
    pipeline: u32,
    buffer: u32,
    x: u32,
    y: u32,
    z: u32,
) -> i32 {
    let data = ctx.data();
    let (Some(pipeline), Some(buffer)) = (
        data.gpu.pipelines.get(&pipeline),
        data.gpu.buffers.get(&buffer),
    ) else {
        return ERR_HANDLE;
    };

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = data
        .gpu
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("guest-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

    let mut encoder = data
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("guest-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }
    data.gpu.queue.submit(Some(encoder.finish()));
    SUCCESS
}

fn pipeline_release(mut ctx: FunctionEnvMut<GpuBridgeEnv>, handle: u32) -> i32 {
    match ctx.data_mut().gpu.pipelines.remove(&handle) {
        Some(_) => SUCCESS,
        None => ERR_HANDLE,
    }
}

// ---------------------------------------------------------------------------
// The model: single-feature linear regression, y = w*x + b, trained by
// full-batch gradient descent on mean squared error. Everything the guest's
// two kernels touch lives in one flat f32 array (the bridge only binds one
// storage buffer per dispatch), laid out as:
//
//   [0]         w
//   [1]         b
//   [2..10)     x[8]      (input features)
//   [10..18)    y[8]      (targets)
//   [18]        loss      (written by the forward/gradient kernel)
//   [19]        dw
//   [20]        db
// ---------------------------------------------------------------------------

const N: usize = 8;
const OFF_W: usize = 0;
const OFF_B: usize = 1;
const OFF_X: usize = OFF_B + 1;
const OFF_Y: usize = OFF_X + N;
const OFF_LOSS: usize = OFF_Y + N;
const OFF_DW: usize = OFF_LOSS + 1;
const OFF_DB: usize = OFF_DW + 1;
const STATE_LEN: usize = OFF_DB + 1;

const LEARNING_RATE: f32 = 0.3;
const TRAINING_STEPS: i32 = 500;
const TRUE_W: f32 = 3.0;
const TRUE_B: f32 = 2.0;

fn mse_loss(w: f32, b: f32, xs: &[f32], ys: &[f32]) -> f32 {
    xs.iter()
        .zip(ys)
        .map(|(&x, &y)| {
            let err = (w * x + b) - y;
            err * err
        })
        .sum::<f32>()
        / xs.len() as f32
}

fn main() -> Result<()> {
    // Forward pass + gradient computation, merged into one kernel: with a
    // batch this small a single GPU invocation just loops over all samples,
    // no reduction/atomics needed.
    let forward_backward_shader = format!(
        r#"
@group(0) @binding(0)
var<storage, read_write> state: array<f32>;

@compute @workgroup_size(1)
fn main() {{
    let w = state[{off_w}u];
    let b = state[{off_b}u];
    var loss = 0.0;
    var dw = 0.0;
    var db = 0.0;
    for (var i: u32 = 0u; i < {n}u; i = i + 1u) {{
        let x = state[{off_x}u + i];
        let y = state[{off_y}u + i];
        let pred = w * x + b;
        let err = pred - y;
        loss = loss + err * err;
        dw = dw + 2.0 * err * x;
        db = db + 2.0 * err;
    }}
    state[{off_loss}u] = loss / f32({n}u);
    state[{off_dw}u] = dw / f32({n}u);
    state[{off_db}u] = db / f32({n}u);
}}
"#,
        off_w = OFF_W,
        off_b = OFF_B,
        off_x = OFF_X,
        off_y = OFF_Y,
        off_loss = OFF_LOSS,
        off_dw = OFF_DW,
        off_db = OFF_DB,
        n = N,
    );

    let update_shader = format!(
        r#"
@group(0) @binding(0)
var<storage, read_write> state: array<f32>;

@compute @workgroup_size(1)
fn main() {{
    state[{off_w}u] = state[{off_w}u] - {lr} * state[{off_dw}u];
    state[{off_b}u] = state[{off_b}u] - {lr} * state[{off_db}u];
}}
"#,
        off_w = OFF_W,
        off_b = OFF_B,
        off_dw = OFF_DW,
        off_db = OFF_DB,
        lr = LEARNING_RATE,
    );

    let fwd_offset = 1024i32;
    let fwd_len = forward_backward_shader.len() as i32;
    let fwd_escaped = forward_backward_shader.replace('\n', "\\n");

    let upd_offset = 2048i32;
    let upd_len = update_shader.len() as i32;
    let upd_escaped = update_shader.replace('\n', "\\n");

    // The guest owns the training loop: upload the initial state once, then
    // dispatch forward+gradient and update on the GPU for every step with no
    // host round trip in between, and only read the result back at the end.
    let wat = format!(
        r#"(module
  (import "env" "buffer_upload" (func $buffer_upload (param i32 i32) (result i32)))
  (import "env" "buffer_read" (func $buffer_read (param i32 i32 i32) (result i32)))
  (import "env" "buffer_write" (func $buffer_write (param i32 i32 i32 i32) (result i32)))
  (import "env" "buffer_release" (func $buffer_release (param i32) (result i32)))
  (import "env" "pipeline_create" (func $pipeline_create (param i32 i32) (result i32)))
  (import "env" "pipeline_dispatch" (func $pipeline_dispatch (param i32 i32 i32 i32 i32) (result i32)))
  (import "env" "pipeline_release" (func $pipeline_release (param i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const {fwd_offset}) "{fwd_escaped}")
  (data (i32.const {upd_offset}) "{upd_escaped}")
  (func (export "run") (param $ptr i32) (param $len i32)
    (local $buf i32)
    (local $fwd i32)
    (local $upd i32)
    (local $step i32)
    (local.set $buf (call $buffer_upload (local.get $ptr) (local.get $len)))
    (local.set $fwd (call $pipeline_create (i32.const {fwd_offset}) (i32.const {fwd_len})))
    (local.set $upd (call $pipeline_create (i32.const {upd_offset}) (i32.const {upd_len})))
    (local.set $step (i32.const 0))
    (block $done
      (loop $continue
        (br_if $done (i32.ge_s (local.get $step) (i32.const {steps})))
        (drop (call $pipeline_dispatch (local.get $fwd) (local.get $buf) (i32.const 1) (i32.const 1) (i32.const 1)))
        (drop (call $pipeline_dispatch (local.get $upd) (local.get $buf) (i32.const 1) (i32.const 1) (i32.const 1)))
        (local.set $step (i32.add (local.get $step) (i32.const 1)))
        (br $continue)))
    (drop (call $buffer_read (local.get $buf) (local.get $ptr) (local.get $len)))
    (drop (call $pipeline_release (local.get $fwd)))
    (drop (call $pipeline_release (local.get $upd)))
    (drop (call $buffer_release (local.get $buf)))))
"#,
        steps = TRAINING_STEPS,
    );
    let wasm_bytes = wat2wasm(wat.as_bytes())?;

    let mut store = Store::default();
    let module = Module::new(&store, wasm_bytes)?;

    let gpu = pollster::block_on(GpuState::new());
    let function_env = FunctionEnv::new(&mut store, GpuBridgeEnv { memory: None, gpu });
    let import_object = imports! {
        "env" => {
            "buffer_upload" => Function::new_typed_with_env(&mut store, &function_env, buffer_upload),
            "buffer_read" => Function::new_typed_with_env(&mut store, &function_env, buffer_read),
            "buffer_write" => Function::new_typed_with_env(&mut store, &function_env, buffer_write),
            "buffer_release" => Function::new_typed_with_env(&mut store, &function_env, buffer_release),
            "pipeline_create" => Function::new_typed_with_env(&mut store, &function_env, pipeline_create),
            "pipeline_dispatch" => Function::new_typed_with_env(&mut store, &function_env, pipeline_dispatch),
            "pipeline_release" => Function::new_typed_with_env(&mut store, &function_env, pipeline_release),
        }
    };

    let instance = Instance::new(&mut store, &module, &import_object)?;
    let memory = instance.exports.get_memory("memory")?;
    function_env.as_mut(&mut store).set_memory(memory.clone());

    // Synthetic dataset: y = 3x + 2, exactly, over 8 points spanning [-1, 1].
    let xs: Vec<f32> = (0..N)
        .map(|i| -1.0 + 2.0 * i as f32 / (N - 1) as f32)
        .collect();
    let ys: Vec<f32> = xs.iter().map(|&x| TRUE_W * x + TRUE_B).collect();

    // Deliberately bad initial guess.
    let (w_init, b_init) = (0.0f32, 0.0f32);
    let initial_loss = mse_loss(w_init, b_init, &xs, &ys);

    let mut state = vec![0.0f32; STATE_LEN];
    state[OFF_W] = w_init;
    state[OFF_B] = b_init;
    state[OFF_X..OFF_X + N].copy_from_slice(&xs);
    state[OFF_Y..OFF_Y + N].copy_from_slice(&ys);

    let ptr = 0u32;
    let byte_len = (STATE_LEN * size_of::<f32>()) as i32;
    memory
        .view(&store)
        .write(ptr as u64, &f32_to_bytes(&state))?;

    let run: TypedFunction<(i32, i32), ()> = instance.exports.get_function("run")?.typed(&store)?;
    run.call(&mut store, ptr as i32, byte_len)?;

    let mut result_bytes = vec![0u8; STATE_LEN * size_of::<f32>()];
    memory.view(&store).read(ptr as u64, &mut result_bytes)?;
    let result = bytes_to_f32(&result_bytes);
    let (final_w, final_b, final_loss) = (result[OFF_W], result[OFF_B], result[OFF_LOSS]);

    println!(
        "guest-owned training: {TRAINING_STEPS} GPU-resident gradient-descent steps, dispatched entirely from wasm"
    );
    println!("  target:  w={TRUE_W}, b={TRUE_B}");
    println!("  initial: w={w_init}, b={b_init}, loss={initial_loss:.6}");
    println!("  learned: w={final_w:.4}, b={final_b:.4}, loss={final_loss:.6}");

    assert!(
        final_loss < initial_loss * 0.05,
        "loss did not converge: {final_loss} vs initial {initial_loss}"
    );
    assert!(
        (final_w - TRUE_W).abs() < 0.1 && (final_b - TRUE_B).abs() < 0.1,
        "learned params too far from truth: w={final_w}, b={final_b}"
    );

    Ok(())
}

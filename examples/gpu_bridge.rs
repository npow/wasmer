//! This example shows that an embedder can bridge a WASM guest to a real
//! GPU *today*, with zero runtime changes, purely by defining custom
//! host-imported functions.
//!
//! The host exposes seven imported functions under the `env` namespace:
//! `buffer_upload`/`buffer_read`/`buffer_write`/`buffer_release` manage
//! GPU-resident storage buffers, and `pipeline_create`/`pipeline_dispatch`/
//! `pipeline_release` manage compute pipelines built from guest-supplied
//! WGSL source. Every buffer and pipeline is addressed by an opaque `u32`
//! handle the guest holds onto across calls, so the guest owns its GPU
//! resources and can drive an arbitrary compute graph over multiple calls
//! instead of one hardcoded operation. All of the interesting work still
//! happens on the host side of the (ptr, len) boundary, via the `wgpu`
//! crate.
//!
//! You can run the example directly by executing in the Wasmer root:
//!
//! ```shell
//! cargo run --example gpu-bridge --release --features "cranelift,gpu-bridge-example"
//! ```
//!
//! Ready?

use anyhow::Result;
use std::collections::HashMap;
use wasmer::{
    AsStoreRef, Function, FunctionEnv, FunctionEnvMut, Instance, Memory, MemoryView, Module, Store,
    TypedFunction, imports, wat2wasm,
};
use wgpu::util::DeviceExt;

/// Doubles every element of a storage buffer of `f32`. Demo kernel supplied
/// by the guest at run time via `pipeline_create` — the host never sees this
/// text except as bytes read out of guest memory.
const DOUBLE_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&data)) {
        data[id.x] = data[id.x] * 2.0;
    }
}
"#;

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

/// Host import: uploads `len` bytes from guest memory at `ptr` into a new
/// GPU-resident storage buffer. Returns the new buffer's handle.
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

/// Host import: reads `len` bytes back from buffer `handle` into guest
/// memory at `ptr`, round-tripping through a mappable staging buffer.
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

/// Host import: writes `len` bytes from guest memory at `ptr` into buffer
/// `handle` starting at `offset`, with no GPU->CPU round trip.
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

/// Host import: releases buffer `handle`. The handle is invalid afterward.
fn buffer_release(mut ctx: FunctionEnvMut<GpuBridgeEnv>, handle: u32) -> i32 {
    match ctx.data_mut().gpu.buffers.remove(&handle) {
        Some(_) => SUCCESS,
        None => ERR_HANDLE,
    }
}

/// Host import: compiles `len` bytes of guest-supplied WGSL source at `ptr`
/// into a compute pipeline. The pipeline's fixed binding contract is one
/// read-write storage buffer at `@group(0) @binding(0)` and an entry point
/// named `main`. Returns the new pipeline's handle.
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

/// Host import: dispatches pipeline `pipeline` against buffer `buffer` with
/// `(x, y, z)` workgroups.
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

/// Host import: releases pipeline `handle`. The handle is invalid afterward.
fn pipeline_release(mut ctx: FunctionEnvMut<GpuBridgeEnv>, handle: u32) -> i32 {
    match ctx.data_mut().gpu.pipelines.remove(&handle) {
        Some(_) => SUCCESS,
        None => ERR_HANDLE,
    }
}

fn main() -> Result<()> {
    // The guest owns the compute graph: it uploads its input, creates a
    // pipeline from WGSL it embeds itself, dispatches it, reads the result
    // back, then releases both handles. The host never hardcodes what the
    // guest computes.
    let shader_offset = 1024i32;
    let shader_len = DOUBLE_SHADER.len() as i32;
    let shader_wat_escaped = DOUBLE_SHADER.replace('\n', "\\n");

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
  (data (i32.const {shader_offset}) "{shader_wat_escaped}")
  (func (export "run") (param $ptr i32) (param $len i32)
    (local $buf i32)
    (local $pipe i32)
    (local.set $buf (call $buffer_upload (local.get $ptr) (local.get $len)))
    (local.set $pipe (call $pipeline_create (i32.const {shader_offset}) (i32.const {shader_len})))
    (drop (call $pipeline_dispatch (local.get $pipe) (local.get $buf) (i32.const 1) (i32.const 1) (i32.const 1)))
    (drop (call $buffer_read (local.get $buf) (local.get $ptr) (local.get $len)))
    (drop (call $pipeline_release (local.get $pipe)))
    (drop (call $buffer_release (local.get $buf)))))
"#
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

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ptr = 0u32;
    let byte_len = (input.len() * size_of::<f32>()) as i32;
    memory
        .view(&store)
        .write(ptr as u64, &f32_to_bytes(&input))?;

    let run: TypedFunction<(i32, i32), ()> = instance.exports.get_function("run")?.typed(&store)?;
    run.call(&mut store, ptr as i32, byte_len)?;

    let mut result_bytes = vec![0u8; input.len() * size_of::<f32>()];
    memory.view(&store).read(ptr as u64, &mut result_bytes)?;
    let output = bytes_to_f32(&result_bytes);

    println!("input: {input:?} -> output: {output:?}");
    assert_eq!(output, input.iter().map(|v| v * 2.0).collect::<Vec<_>>());

    Ok(())
}

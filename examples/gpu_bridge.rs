//! This example shows that an embedder can bridge a WASM guest to a real
//! GPU *today*, with zero runtime changes, purely by defining a custom
//! host-imported function.
//!
//! The host exposes one imported function, `gpu_double`, that takes a
//! `(ptr, len)` pointer into the guest's own linear memory, reads out an
//! array of `f32`, doubles every element with a real GPU compute-shader
//! dispatch (via the `wgpu` crate), and writes the result back into that
//! same memory region. The guest module just forwards its `run` export to
//! that import, so all of the interesting work happens on the host side of
//! the (ptr, len) boundary.
//!
//! You can run the example directly by executing in the Wasmer root:
//!
//! ```shell
//! cargo run --example gpu-bridge --release --features "cranelift,gpu-bridge-example"
//! ```
//!
//! Ready?

use anyhow::Result;
use wasmer::{
    AsStoreRef, Function, FunctionEnv, FunctionEnvMut, Instance, Memory, MemoryView, Module, Store,
    TypedFunction, imports, wat2wasm,
};
use wgpu::util::DeviceExt;

/// Doubles every element of a storage buffer of `f32`.
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

/// Runs `data` through a doubling compute shader on the GPU and returns the
/// result. Sets up (and tears down) its own `wgpu` instance, so it's only
/// meant to be called a handful of times, not in a hot loop.
async fn double_on_gpu(data: &[f32]) -> Vec<f32> {
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

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("double"),
        source: wgpu::ShaderSource::Wgsl(DOUBLE_SHADER.into()),
    });

    let byte_len = std::mem::size_of_val(data) as wgpu::BufferAddress;

    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("data"),
        contents: &f32_to_bytes(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("double_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("double_bind_group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("double_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (data.len() as u32).div_ceil(64).max(1);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, byte_len);
    queue.submit(Some(encoder.finish()));

    // Round-trip the result back to the CPU via a mappable staging buffer.
    let slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("map_async receiver dropped");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
    rx.recv()
        .expect("map_async callback never ran")
        .expect("failed to map staging buffer");

    let output = bytes_to_f32(&slice.get_mapped_range().expect("no mapped range"));
    staging_buffer.unmap();
    output
}

/// Environment shared with the `gpu_double` host function: just enough to
/// reach the guest's exported memory.
pub struct GpuBridgeEnv {
    memory: Option<Memory>,
}

impl GpuBridgeEnv {
    fn set_memory(&mut self, memory: Memory) {
        self.memory = Some(memory);
    }

    fn view<'a>(&'a self, store: &'a impl AsStoreRef) -> MemoryView<'a> {
        self.memory.as_ref().unwrap().view(store)
    }
}

/// Host import: reads `len` `f32`s from guest memory at `ptr`, doubles them
/// on the GPU, and writes the result back into the same region.
fn gpu_double(ctx: FunctionEnvMut<GpuBridgeEnv>, ptr: u32, len: u32) {
    let mut bytes = vec![0u8; len as usize * size_of::<f32>()];
    ctx.data().view(&ctx).read(ptr as u64, &mut bytes).unwrap();

    let output = pollster::block_on(double_on_gpu(&bytes_to_f32(&bytes)));

    ctx.data()
        .view(&ctx)
        .write(ptr as u64, &f32_to_bytes(&output))
        .unwrap();
}

fn main() -> Result<()> {
    let wasm_bytes = wat2wasm(
        br#"
(module
  (import "env" "gpu_double" (func $gpu_double (param i32 i32)))
  (memory (export "memory") 1)
  (func (export "run") (param $ptr i32) (param $len i32)
    local.get $ptr
    local.get $len
    call $gpu_double))
"#,
    )?;

    let mut store = Store::default();
    let module = Module::new(&store, wasm_bytes)?;

    let function_env = FunctionEnv::new(&mut store, GpuBridgeEnv { memory: None });
    let import_object = imports! {
        "env" => {
            "gpu_double" => Function::new_typed_with_env(&mut store, &function_env, gpu_double),
        }
    };

    let instance = Instance::new(&mut store, &module, &import_object)?;
    let memory = instance.exports.get_memory("memory")?;
    function_env.as_mut(&mut store).set_memory(memory.clone());

    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ptr = 0u32;
    memory
        .view(&store)
        .write(ptr as u64, &f32_to_bytes(&input))?;

    let run: TypedFunction<(i32, i32), ()> = instance.exports.get_function("run")?.typed(&store)?;
    run.call(&mut store, ptr as i32, input.len() as i32)?;

    let mut result_bytes = vec![0u8; input.len() * size_of::<f32>()];
    memory.view(&store).read(ptr as u64, &mut result_bytes)?;
    let output = bytes_to_f32(&result_bytes);

    println!("input: {input:?} -> output: {output:?}");
    assert_eq!(output, input.iter().map(|v| v * 2.0).collect::<Vec<_>>());

    Ok(())
}

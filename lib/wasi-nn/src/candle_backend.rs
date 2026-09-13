//! A reference [`NnBackend`] backed by the `candle` crate. Runs a genuinely
//! real computation (not a mock) on CPU or CUDA, over an arbitrary layer
//! stack -- not one hardcoded architecture.
//!
//! ## Graph format
//!
//! wasi-nn's `graph_encoding` enum (openvino/onnx/tensorflow/pytorch/
//! tensorflowlite/autodetect) has no variant for "a candle-native model", and
//! candle itself doesn't parse ONNX/OpenVINO/TensorFlow/TorchScript graphs.
//! This backend therefore only accepts `GraphEncoding::Autodetect`, and
//! interprets the two-blob `$graph_builder_array` as:
//!
//!   - blob 0: a JSON [`ModelConfig`] -- an ordered list of layers, each
//!     naming the safetensors tensors it needs (see [`LayerConfig`]).
//!   - blob 1: a safetensors file holding those named tensors.
//!
//! This mirrors how real wasi-nn backends split "structure" from "weights"
//! (e.g. OpenVINO's `model.xml` + `model.bin`), and is exactly the shape a
//! model exported from PyTorch (`torch.nn.Module.state_dict()` ->
//! `safetensors.torch.save_file`, plus a small hand-written layer list) takes.
//! `examples/wasi_nn_gpu_demo.rs` does exactly that with a real digit
//! classifier trained in PyTorch (see `examples/assets/digits_mlp/`).
//!
//! This is a deliberate, documented limitation: guest code that calls
//! `wasi_ephemeral_nn` is portable across hosts, but a model built for this
//! reference backend's config format is not interchangeable with an ONNX one.
//! A real ONNX/OpenVINO backend is a separate, much larger `NnBackend` impl
//! that can be swapped in without touching `wasmer-wasix` (or this file).
//!
//! Supported layers today: `linear`, `relu` (see [`LayerConfig`]). Adding a
//! new layer type means adding a variant there and to [`Layer::forward`] --
//! no ABI or `wasmer-wasix` change needed.

use candle_core::{DType, Device, Tensor as CTensor};
use serde::Deserialize;

use crate::backend::{NnBackend, NnExecutionContext, NnGraph};
use crate::types::{ExecutionTarget, GraphEncoding, NnErrno, Tensor, TensorType};

/// The JSON structure of a graph's first builder blob.
#[derive(Debug, Deserialize)]
struct ModelConfig {
    layers: Vec<LayerConfig>,
}

/// One layer, naming the safetensors tensors it reads its parameters from.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum LayerConfig {
    /// `y = x @ weight^T (+ bias)`.
    Linear {
        weight: String,
        bias: Option<String>,
    },
    Relu,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CandleBackend;

impl CandleBackend {
    pub fn new() -> Self {
        Self
    }

    fn device_for(target: ExecutionTarget) -> Result<Device, NnErrno> {
        match target {
            ExecutionTarget::Cpu => Ok(Device::Cpu),
            ExecutionTarget::Gpu => Self::cuda_device(),
            ExecutionTarget::Tpu => Err(NnErrno::UnsupportedOperation),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Result<Device, NnErrno> {
        Device::new_cuda(0).map_err(|_| NnErrno::RuntimeError)
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_device() -> Result<Device, NnErrno> {
        Err(NnErrno::UnsupportedOperation)
    }
}

impl NnBackend for CandleBackend {
    fn load(
        &self,
        builders: &[Vec<u8>],
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Box<dyn NnGraph>, NnErrno> {
        if encoding != GraphEncoding::Autodetect {
            return Err(NnErrno::InvalidEncoding);
        }
        let [config_bytes, weights_bytes] = builders else {
            return Err(NnErrno::InvalidArgument);
        };
        let device = Self::device_for(target)?;

        let config: ModelConfig =
            serde_json::from_slice(config_bytes).map_err(|_| NnErrno::InvalidArgument)?;
        let tensors = candle_core::safetensors::load_buffer(weights_bytes, &device)
            .map_err(|_| NnErrno::RuntimeError)?;

        let mut layers = Vec::with_capacity(config.layers.len());
        for layer in config.layers {
            layers.push(match layer {
                LayerConfig::Linear { weight, bias } => {
                    let weight = tensors.get(&weight).ok_or(NnErrno::RuntimeError)?.clone();
                    let bias = match bias {
                        Some(name) => {
                            Some(tensors.get(&name).ok_or(NnErrno::RuntimeError)?.clone())
                        }
                        None => None,
                    };
                    Layer::Linear { weight, bias }
                }
                LayerConfig::Relu => Layer::Relu,
            });
        }

        Ok(Box::new(LayerStackGraph { layers, device }))
    }
}

/// One executable layer, resolved against its weight tensors.
#[derive(Debug, Clone)]
enum Layer {
    Linear {
        weight: CTensor,
        bias: Option<CTensor>,
    },
    Relu,
}

impl Layer {
    fn forward(&self, x: &CTensor) -> candle_core::Result<CTensor> {
        match self {
            Layer::Linear { weight, bias } => {
                let mut out = x.broadcast_matmul(&weight.t()?)?;
                if let Some(bias) = bias {
                    out = out.broadcast_add(bias)?;
                }
                Ok(out)
            }
            Layer::Relu => x.relu(),
        }
    }
}

#[derive(Debug)]
struct LayerStackGraph {
    layers: Vec<Layer>,
    device: Device,
}

impl NnGraph for LayerStackGraph {
    fn init_execution_context(&self) -> Result<Box<dyn NnExecutionContext>, NnErrno> {
        Ok(Box::new(LayerStackExecutionContext {
            layers: self.layers.clone(),
            device: self.device.clone(),
            input: None,
            output: None,
        }))
    }
}

#[derive(Debug)]
struct LayerStackExecutionContext {
    layers: Vec<Layer>,
    device: Device,
    input: Option<CTensor>,
    output: Option<CTensor>,
}

impl NnExecutionContext for LayerStackExecutionContext {
    fn set_input(&mut self, index: u32, tensor: Tensor) -> Result<(), NnErrno> {
        if index != 0 {
            return Err(NnErrno::InvalidArgument);
        }
        if tensor.ty != TensorType::F32 {
            return Err(NnErrno::UnsupportedOperation);
        }
        let dims: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        let numel: usize = dims.iter().product();
        if tensor.data.len() != numel * 4 {
            return Err(NnErrno::InvalidArgument);
        }
        let floats: Vec<f32> = tensor
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let input =
            CTensor::from_vec(floats, dims, &self.device).map_err(|_| NnErrno::RuntimeError)?;
        self.input = Some(input);
        Ok(())
    }

    fn compute(&mut self) -> Result<(), NnErrno> {
        let input = self.input.as_ref().ok_or(NnErrno::InvalidArgument)?;
        // Every layer here preserves "is this a batch or a bare vector", so
        // unsqueeze/squeeze once at the boundary rather than per layer.
        let is_vector = input.dims().len() == 1;
        let mut x = if is_vector {
            input.unsqueeze(0).map_err(|_| NnErrno::RuntimeError)?
        } else {
            input.clone()
        };
        for layer in &self.layers {
            x = layer.forward(&x).map_err(|_| NnErrno::RuntimeError)?;
        }
        if is_vector {
            x = x.squeeze(0).map_err(|_| NnErrno::RuntimeError)?;
        }
        self.output = Some(x);
        Ok(())
    }

    fn get_output(&self, index: u32) -> Result<Vec<u8>, NnErrno> {
        if index != 0 {
            return Err(NnErrno::InvalidArgument);
        }
        let output = self.output.as_ref().ok_or(NnErrno::InvalidArgument)?;
        let flat = output
            .flatten_all()
            .map_err(|_| NnErrno::RuntimeError)?
            .to_dtype(DType::F32)
            .map_err(|_| NnErrno::RuntimeError)?
            .to_vec1::<f32>()
            .map_err(|_| NnErrno::RuntimeError)?;
        let mut bytes = Vec::with_capacity(flat.len() * 4);
        for f in flat {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// Builds a tiny two-layer (`Linear -> Relu -> Linear`-shaped, minus the
    /// final relu so outputs can go negative) MLP as a `(config_json,
    /// safetensors_bytes)` pair, with no network access and no external model
    /// file -- proves the general layer-stack path, not tied to a specific
    /// trained model. `examples/wasi_nn_gpu_demo.rs` covers the real,
    /// PyTorch-trained case.
    fn tiny_mlp() -> (Vec<u8>, Vec<u8>) {
        let dev = Device::Cpu;
        // layer0: 2 -> 3, weight [[1,0],[0,1],[1,1]], bias [0,0,1]
        let w0 = CTensor::from_vec(vec![1f32, 0., 0., 1., 1., 1.], (3, 2), &dev).unwrap();
        let b0 = CTensor::from_vec(vec![0f32, 0., 1.], 3, &dev).unwrap();
        // layer1 (after relu): 3 -> 1, weight [[1,1,1]], bias [0]
        let w1 = CTensor::from_vec(vec![1f32, 1., 1.], (1, 3), &dev).unwrap();
        let b1 = CTensor::from_vec(vec![0f32], 1, &dev).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("w0".to_string(), w0);
        tensors.insert("b0".to_string(), b0);
        tensors.insert("w1".to_string(), w1);
        tensors.insert("b1".to_string(), b1);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let weights_bytes = std::fs::read(path).unwrap();

        let config = serde_json::json!({
            "layers": [
                {"type": "linear", "weight": "w0", "bias": "b0"},
                {"type": "relu"},
                {"type": "linear", "weight": "w1", "bias": "b1"},
            ]
        });
        let config_bytes = serde_json::to_vec(&config).unwrap();
        (config_bytes, weights_bytes)
    }

    fn run(target: ExecutionTarget, input: [f32; 2]) -> f32 {
        let (config_bytes, weights_bytes) = tiny_mlp();
        let backend = CandleBackend::new();
        let graph = backend
            .load(
                &[config_bytes, weights_bytes],
                GraphEncoding::Autodetect,
                target,
            )
            .expect("load");
        let mut ctx = graph.init_execution_context().expect("init context");

        let tensor = Tensor {
            dimensions: vec![2],
            ty: TensorType::F32,
            data: input.iter().flat_map(|f| f.to_le_bytes()).collect(),
        };
        ctx.set_input(0, tensor).expect("set_input");
        ctx.compute().expect("compute");
        let output = ctx.get_output(0).expect("get_output");
        assert_eq!(output.len(), 4);
        f32::from_le_bytes(output.try_into().unwrap())
    }

    #[test]
    fn runs_a_two_layer_mlp_end_to_end_on_cpu() {
        // [3,4] -> layer0 -> [3, 4, 8] -> relu -> [3, 4, 8] -> layer1 -> [15]
        assert_eq!(run(ExecutionTarget::Cpu, [3.0, 4.0]), 15.0);
    }

    #[test]
    fn relu_actually_clips_negatives() {
        // [-5,-5] -> layer0 -> [-5, -5, -9] -> relu -> [0, 0, 0] -> layer1 -> [0]
        assert_eq!(run(ExecutionTarget::Cpu, [-5.0, -5.0]), 0.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn runs_a_two_layer_mlp_end_to_end_on_cuda() {
        assert_eq!(run(ExecutionTarget::Gpu, [3.0, 4.0]), 15.0);
    }

    #[test]
    fn rejects_non_autodetect_encoding() {
        let backend = CandleBackend::new();
        let err = backend
            .load(&[vec![], vec![]], GraphEncoding::Onnx, ExecutionTarget::Cpu)
            .unwrap_err();
        assert_eq!(err, NnErrno::InvalidEncoding);
    }

    #[test]
    fn rejects_wrong_blob_count() {
        let backend = CandleBackend::new();
        let err = backend
            .load(&[vec![]], GraphEncoding::Autodetect, ExecutionTarget::Cpu)
            .unwrap_err();
        assert_eq!(err, NnErrno::InvalidArgument);
    }
}

//! A reference [`NnBackend`] backed by the `candle` crate. Runs a genuinely
//! real computation (not a mock) on CPU or CUDA.
//!
//! ## Scope
//!
//! wasi-nn's `graph_encoding` enum (openvino/onnx/tensorflow/pytorch/
//! tensorflowlite/autodetect) has no variant for "a candle-native model", and
//! candle itself doesn't parse ONNX/OpenVINO/TensorFlow graphs. This backend
//! therefore only accepts `GraphEncoding::Autodetect`, and interprets the
//! graph_builder bytes as a single safetensors blob holding exactly one
//! linear layer: a `weight` tensor (`[out_features, in_features]`, f32) and an
//! optional `bias` tensor (`[out_features]`, f32). `compute` runs
//! `output = input @ weight^T (+ bias)`.
//!
//! This is a deliberate, documented limitation (see the portability caveat in
//! the design writeup this crate implements): guest code that calls
//! `wasi_ephemeral_nn` is portable across hosts, but a model built for this
//! reference backend is not interchangeable with an ONNX/OpenVINO one. A real
//! ONNX/OpenVINO backend is a separate, much larger `NnBackend` impl that can
//! be swapped in without touching `wasmer-wasix`.

use candle_core::{DType, Device, Tensor as CTensor};

use crate::backend::{NnBackend, NnExecutionContext, NnGraph};
use crate::types::{ExecutionTarget, GraphEncoding, NnErrno, Tensor, TensorType};

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
        let bytes = builders.first().ok_or(NnErrno::InvalidArgument)?;
        let device = Self::device_for(target)?;

        let tensors = candle_core::safetensors::load_buffer(bytes, &device)
            .map_err(|_| NnErrno::RuntimeError)?;
        let weight = tensors.get("weight").ok_or(NnErrno::RuntimeError)?.clone();
        let bias = tensors.get("bias").cloned();

        Ok(Box::new(LinearGraph {
            weight,
            bias,
            device,
        }))
    }
}

#[derive(Debug)]
struct LinearGraph {
    weight: CTensor,
    bias: Option<CTensor>,
    device: Device,
}

impl NnGraph for LinearGraph {
    fn init_execution_context(&self) -> Result<Box<dyn NnExecutionContext>, NnErrno> {
        Ok(Box::new(LinearExecutionContext {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            device: self.device.clone(),
            input: None,
            output: None,
        }))
    }
}

#[derive(Debug)]
struct LinearExecutionContext {
    weight: CTensor,
    bias: Option<CTensor>,
    device: Device,
    input: Option<CTensor>,
    output: Option<CTensor>,
}

impl NnExecutionContext for LinearExecutionContext {
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
        let weight_t = self.weight.t().map_err(|_| NnErrno::RuntimeError)?;
        // `broadcast_matmul` needs rank >= 2; treat a bare feature vector as a
        // batch of one row, then undo that afterward.
        let is_vector = input.dims().len() == 1;
        let input2d = if is_vector {
            input.unsqueeze(0).map_err(|_| NnErrno::RuntimeError)?
        } else {
            input.clone()
        };
        let mut output = input2d
            .broadcast_matmul(&weight_t)
            .map_err(|_| NnErrno::RuntimeError)?;
        if let Some(bias) = &self.bias {
            output = output
                .broadcast_add(bias)
                .map_err(|_| NnErrno::RuntimeError)?;
        }
        if is_vector {
            output = output.squeeze(0).map_err(|_| NnErrno::RuntimeError)?;
        }
        self.output = Some(output);
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

    /// Builds a tiny `y = x @ W^T + b` safetensors blob (2 in, 3 out, fixed
    /// weights) with no network access and no external model file, then runs
    /// it end to end through the public `NnBackend` trait -- this is real
    /// candle inference, not a mock, just on a hand-picked model.
    fn tiny_linear_safetensors() -> Vec<u8> {
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
        std::fs::read(path).unwrap()
    }

    #[test]
    fn runs_real_cpu_inference_end_to_end() {
        let bytes = tiny_linear_safetensors();
        let backend = CandleBackend::new();
        let graph = backend
            .load(&[bytes], GraphEncoding::Autodetect, ExecutionTarget::Cpu)
            .expect("load");
        let mut ctx = graph.init_execution_context().expect("init context");

        let input = Tensor {
            dimensions: vec![2],
            ty: TensorType::F32,
            data: [3f32, 4f32].iter().flat_map(|f| f.to_le_bytes()).collect(),
        };
        ctx.set_input(0, input).expect("set_input");
        ctx.compute().expect("compute");
        let output = ctx.get_output(0).expect("get_output");

        let floats: Vec<f32> = output
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // [3,4] @ [[1,0],[0,1],[1,1]]^T + [0,0,1] = [3, 4, 3+4+1] = [3, 4, 8]
        assert_eq!(floats, vec![3.0, 4.0, 8.0]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn runs_real_cuda_inference_end_to_end() {
        let bytes = tiny_linear_safetensors();
        let backend = CandleBackend::new();
        let graph = backend
            .load(&[bytes], GraphEncoding::Autodetect, ExecutionTarget::Gpu)
            .expect("load on GPU");
        let mut ctx = graph.init_execution_context().expect("init context");

        let input = Tensor {
            dimensions: vec![2],
            ty: TensorType::F32,
            data: [3f32, 4f32].iter().flat_map(|f| f.to_le_bytes()).collect(),
        };
        ctx.set_input(0, input).expect("set_input");
        ctx.compute().expect("compute on GPU");
        let output = ctx.get_output(0).expect("get_output");

        let floats: Vec<f32> = output
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(floats, vec![3.0, 4.0, 8.0]);
    }

    #[test]
    fn rejects_non_autodetect_encoding() {
        let backend = CandleBackend::new();
        let err = backend
            .load(&[vec![]], GraphEncoding::Onnx, ExecutionTarget::Cpu)
            .unwrap_err();
        assert_eq!(err, NnErrno::InvalidEncoding);
    }
}

//! Wire types for the `wasi_ephemeral_nn` ABI (WASI-NN witx, pre-Component-Model).
//!
//! Discriminants match `wasi-nn.witx` exactly (`crates/wasi-nn/witx/wasi-nn.witx` in
//! `bytecodealliance/wasmtime`) so a guest built against wasmtime-wasi-nn's witx ABI
//! runs unmodified against this host.

/// `$nn_errno` (u16 tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum NnErrno {
    Success = 0,
    InvalidArgument = 1,
    InvalidEncoding = 2,
    MissingMemory = 3,
    Busy = 4,
    RuntimeError = 5,
    UnsupportedOperation = 6,
    TooLarge = 7,
    NotFound = 8,
}

impl NnErrno {
    pub fn to_u32(self) -> u32 {
        self as u16 as u32
    }
}

/// `$graph_encoding` (u8 tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GraphEncoding {
    Openvino = 0,
    Onnx = 1,
    Tensorflow = 2,
    Pytorch = 3,
    Tensorflowlite = 4,
    Autodetect = 5,
}

impl TryFrom<u8> for GraphEncoding {
    type Error = NnErrno;

    fn try_from(v: u8) -> Result<Self, NnErrno> {
        Ok(match v {
            0 => Self::Openvino,
            1 => Self::Onnx,
            2 => Self::Tensorflow,
            3 => Self::Pytorch,
            4 => Self::Tensorflowlite,
            5 => Self::Autodetect,
            _ => return Err(NnErrno::InvalidArgument),
        })
    }
}

/// `$execution_target` (u8 tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExecutionTarget {
    Cpu = 0,
    Gpu = 1,
    Tpu = 2,
}

impl TryFrom<u8> for ExecutionTarget {
    type Error = NnErrno;

    fn try_from(v: u8) -> Result<Self, NnErrno> {
        Ok(match v {
            0 => Self::Cpu,
            1 => Self::Gpu,
            2 => Self::Tpu,
            _ => return Err(NnErrno::InvalidArgument),
        })
    }
}

/// `$tensor_type` (u8 tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorType {
    F16 = 0,
    F32 = 1,
    F64 = 2,
    U8 = 3,
    I32 = 4,
    I64 = 5,
}

impl TryFrom<u8> for TensorType {
    type Error = NnErrno;

    fn try_from(v: u8) -> Result<Self, NnErrno> {
        Ok(match v {
            0 => Self::F16,
            1 => Self::F32,
            2 => Self::F64,
            3 => Self::U8,
            4 => Self::I32,
            5 => Self::I64,
            _ => return Err(NnErrno::InvalidArgument),
        })
    }
}

/// `$tensor`: dimensions + element type + raw little-endian element bytes.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub dimensions: Vec<u32>,
    pub ty: TensorType,
    pub data: Vec<u8>,
}

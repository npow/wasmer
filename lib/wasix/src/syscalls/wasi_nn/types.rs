//! Wasm-memory record layouts for the `wasi_ephemeral_nn` witx ABI. Field order
//! matches `wasi-nn.witx` declaration order; `#[repr(C)]` gives the same
//! natural-alignment layout the witx record-layout algorithm specifies (and
//! that `wasmer::ValueType`'s derive knows how to zero-pad).
//!
//! This ABI is 32-bit-pointer only (witx has no `Memory64` variant), unlike the
//! rest of WASIX which is generic over [`wasmer::MemorySize`].

use wasmer::ValueType;

/// `$graph_builder`: one raw model byte blob, as a `(ptr, len)` pair into guest
/// memory. `$graph_builder_array` is `builder_ptr`+`builder_len` many of these.
#[repr(C)]
#[derive(Debug, Clone, Copy, ValueType)]
pub struct WasmGraphBuilder {
    pub ptr: u32,
    pub len: u32,
}

/// `$tensor`: `{ dimensions: list<u32>, type: tensor_type, data: list<u8> }`.
#[repr(C)]
#[derive(Debug, Clone, Copy, ValueType)]
pub struct WasmTensor {
    pub dimensions_ptr: u32,
    pub dimensions_len: u32,
    pub ty: u8,
    pub data_ptr: u32,
    pub data_len: u32,
}

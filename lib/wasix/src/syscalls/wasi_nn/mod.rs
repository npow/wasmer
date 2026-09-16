//! Host functions for the `wasi_ephemeral_nn` import namespace (WASI-NN's
//! pre-Component-Model witx ABI; see `wasi-nn.witx` in
//! `bytecodealliance/wasmtime`). 32-bit-pointer only, unlike the rest of WASIX.
//!
//! Gated by this crate's `wasi-nn` feature and, at every call site, by
//! [`crate::capabilities::CapabilityNnV1`].

mod compute;
mod get_output;
mod init_execution_context;
mod load;
mod load_by_name;
mod set_input;
mod types;

pub use compute::*;
pub use get_output::*;
pub use init_execution_context::*;
pub use load::*;
pub use load_by_name::*;
pub use set_input::*;

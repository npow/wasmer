//! Phase 1 of the `wasi_webgpu_v0` async spike: the Component Model async
//! ABI's own documented minimal single-active-task-at-a-time subset --
//! `task.return`-equivalent completion, `waitable-set.new`, `waitable.join`,
//! `waitable-set.wait`, and subtask cleanup -- real enough to be the actual
//! foundation a wasi:webgpu bridge's `request-adapter`/`map-async`/etc. would
//! sit on, still deliberately narrow: no `stream`/`future` built-ins, no
//! `backpressure.*`/`context.*`/`thread.*`, and (Phase 1's specific
//! boundary) at most one joined waitable per waitable set -- multiple
//! concurrently in-flight subtasks per set is Phase 2's job, not this one's.
//!
//! This graduates Phase 0's single fake `request_adapter_spike` import (see
//! that phase's now-removed `wasi_webgpu_spike.rs`) into five imports that
//! actually separate "start an async operation" from "suspend until it
//! resolves" -- the real shape any async Component Model export needs, since
//! `request-adapter` itself must return control to the caller so it can join
//! the resulting subtask into a waitable set before blocking on it.
//!
//! This is still NOT wasi:webgpu. It implements none of that WIT interface's
//! actual method names, its canonical ABI lowering, or the Component Model
//! itself -- it is a hand-written wasix host-import ABI (core wasm, no
//! `canon lower`) that proves the *mechanism* a real bridge would need.
//!
//! Design (state machine): see `crate::state::WebgpuState`'s doc comment for
//! the subtask/waitable-set representation and how suspend/resume is woken.
//! The one real async host import, [`waitable_set_wait`], suspends the
//! guest's coroutine via the same `Function::new_typed_with_env_async` /
//! `AsyncFunctionEnvMut` primitive Phase 0 proved (see `context_switch` for
//! where Wasmer got it from) -- it holds the WasiEnv/store write lock only
//! for the two brief synchronous edges (fetching the shared state at the
//! start, writing the resolved payload into guest memory at the end), never
//! across the actual suspend, matching Phase 0's own discipline.
//!
//! Gated by this crate's `wasi-webgpu-concurrency` feature and, at every call
//! site, by [`crate::capabilities::CapabilityWebgpuSpikeV1`].

mod request_adapter_start;
mod subtask_drop;
mod waitable_join;
mod waitable_set_new;
mod waitable_set_wait;

pub use request_adapter_start::*;
pub use subtask_drop::*;
pub use waitable_join::*;
pub use waitable_set_new::*;
pub use waitable_set_wait::*;

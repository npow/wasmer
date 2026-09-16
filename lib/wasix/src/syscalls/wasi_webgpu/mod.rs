//! Phase 1 (now Phase 2a) of the `wasi_webgpu_v0` async spike: the Component
//! Model async ABI's own documented minimal single-active-task-at-a-time
//! subset -- `task.return`-equivalent completion, `waitable-set.new`,
//! `waitable.join`, `waitable-set.wait`, and subtask cleanup -- real enough
//! to be the actual foundation a wasi:webgpu bridge's
//! `request-adapter`/`map-async`/etc. would sit on, still deliberately
//! narrow: no `stream`/`future` built-ins, no `backpressure.*`/`context.*`/
//! `thread.*`, and no real multi-task reentrancy (each guest instance still
//! has at most one *top-level* pending wait at a time; see below for what
//! Phase 2a actually adds within that).
//!
//! This graduates Phase 0's single fake `request_adapter_spike` import (see
//! that phase's now-removed `wasi_webgpu_spike.rs`) into five imports that
//! actually separate "start an async operation" from "suspend until it
//! resolves" -- the real shape any async Component Model export needs, since
//! `request-adapter` itself must return control to the caller so it can join
//! the resulting subtask into a waitable set before blocking on it.
//!
//! **Phase 2a**, on top of Phase 1's exactly-one-waitable-per-set: a
//! waitable set now holds up to `MAX_WAITABLES_PER_SET` joined subtasks, and
//! [`waitable_set_wait`] returns whichever one resolves *first* -- fan-in
//! within a single wait, not concurrent top-level calls (that sidestep
//! already exists today via WASIX thread-spawn, each thread getting its own
//! `Store`; see this phase's scoping discussion). `request_adapter_start`
//! also gained a `delay_ms` parameter so tests can stagger completion order
//! deterministically -- a test/demo knob, not part of any real ABI this
//! stands in for.
//!
//! **`future.*`**, on top of Phase 2a: [`future_new`]/[`future_read`]/
//! [`future_drop`] add a second waitable kind alongside subtasks -- a
//! future is a one-shot value, order-enforced (`future_read` returns
//! immediately if a value is already there) or suspend-based (returns
//! [`crate::state::WebgpuErrno::Blocked`], then the guest joins it into a
//! waitable set exactly like a subtask). No separate `[future-writer]`/
//! `[future-reader]` resource-pair split like the real spec -- one handle,
//! matching this crate's existing single-handle convention. No guest-facing
//! `future_write` either: this bridge's only realistic producer of a
//! future's value is a fake async host/GPU operation, so
//! [`future_resolve_after`] (mirroring `request_adapter_start`'s spawn-a-
//! background-task shape, decoupled from allocation) fills that role
//! instead. `stream.*` is deliberately left for a follow-up once this
//! pattern is proven; `backpressure.*`/`context.*`/`thread.*` remain out of
//! scope, per this session's own Phase 2 research, until real multi-task
//! reentrancy exists.
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
//! for the brief synchronous edges (fetching the shared state at the start,
//! writing the resolved handle+payload into guest memory at the end), never
//! across the actual suspend, matching Phase 0's own discipline.
//!
//! Gated by this crate's `wasi-webgpu-concurrency` feature and, at every call
//! site, by [`crate::capabilities::CapabilityWebgpuSpikeV1`].

mod future_drop;
mod future_new;
mod future_read;
mod future_resolve_after;
mod request_adapter_start;
mod subtask_drop;
mod waitable_join;
mod waitable_set_new;
mod waitable_set_wait;

pub use future_drop::*;
pub use future_new::*;
pub use future_read::*;
pub use future_resolve_after::*;
pub use request_adapter_start::*;
pub use subtask_drop::*;
pub use waitable_join::*;
pub use waitable_set_new::*;
pub use waitable_set_wait::*;

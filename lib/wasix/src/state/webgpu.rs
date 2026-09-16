//! Handle tables for Phase 1 of the `wasi_webgpu_v0` async spike -- see
//! `syscalls/wasi_webgpu/` for the host imports built on top of this.
//!
//! Lives on [`super::WasiState`] (the `Arc`-shared, per-process state), not
//! `WasiEnv` (cloned per thread) -- a subtask started by one thread must be
//! visible to and waitable-on by every other thread of the same process,
//! exactly like `WasiState::nn`.
//!
//! Handles do not survive `fork`, snapshot, or restore: each of those
//! rebuilds a fresh, empty [`WebgpuState`], exactly like `NnState`.
//!
//! Design: a subtask is [`SubtaskSlot::Pending`] (holding an `Arc<Notify>`)
//! while its background host future runs, or [`SubtaskSlot::Resolved`] (with
//! its `u32` result payload) once that future completes. The spawning task
//! transitions the entry under this state's lock and calls
//! `Notify::notify_one()`; `Notify`'s single stored permit means a
//! `waitable_set_wait` caller that checks-then-awaits never misses the
//! wakeup, even if resolution races the check-and-drop-lock step.
//!
//! Phase 2a: a waitable set holds up to [`MAX_WAITABLES_PER_SET`] joined
//! subtask handles (`Vec<u32>`, was `Option<u32>` in Phase 1's
//! exactly-one-per-set scope). `waitable_set_wait` returns whichever member
//! resolves *first*, then removes it from the set -- a subtask's resolution
//! is a one-shot terminal event, so once delivered it is no longer
//! monitored; the guest separately calls `subtask_drop` to free the handle
//! itself. The cap exists for the same reason `examples/gpu_bridge.rs`'s
//! `SessionLimits` does: a bounded per-instance resource, not an unbounded
//! one a hostile guest could grow without limit.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Notify;

/// Bounded fan-in per waitable set (Phase 2a's scope: N joined subtasks,
/// still no real multi-task reentrancy -- see `syscalls/wasi_webgpu/mod.rs`).
pub(crate) const MAX_WAITABLES_PER_SET: usize = 8;

/// Wire-level status codes for `wasi_webgpu_v0`'s Phase 1 host imports.
/// Modeled after `wasmer_wasi_nn::NnErrno`'s discriminant-per-condition
/// style, but local to this crate: this ABI has no separate published crate
/// and never will, since it is explicitly not a real `wasi:webgpu`
/// implementation (see `syscalls/wasi_webgpu/mod.rs`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WebgpuErrno {
    Success = 0,
    /// The engine does not support async execution (mirrors
    /// `context_switch_not_supported`'s condition).
    Unsupported = -1,
    /// Denied by `Capabilities::webgpu_spike.allow`.
    CapabilityDenied = -2,
    /// A pointer argument fell outside guest memory.
    MissingMemory = -3,
    /// The handle was never issued, or was already released.
    BadHandle = -4,
    /// `waitable_join` targeted a set already at [`MAX_WAITABLES_PER_SET`]
    /// members.
    SetOccupied = -5,
    /// `waitable_set_wait` targeted a set with no joined waitable.
    SetEmpty = -6,
    /// This instance's 32-bit handle counter is exhausted.
    TooManyHandles = -7,
}

impl WebgpuErrno {
    pub fn to_i32(self) -> i32 {
        self as i32
    }
}

#[derive(Debug)]
pub(crate) enum SubtaskSlot {
    Pending(Arc<Notify>),
    Resolved(u32),
}

/// Outcome of a non-blocking look-up of a waitable set's joined subtasks.
pub(crate) enum WaitOutcome {
    /// One joined subtask has already resolved -- its handle and payload.
    /// No need to await anything.
    Ready(u32, u32),
    /// All joined subtasks are still pending -- await any one of these
    /// (handle, `Notify`) pairs before re-polling. Never empty: only
    /// produced when the set has at least one member and none are resolved.
    Pending(Vec<(u32, Arc<Notify>)>),
}

#[derive(Default, Debug)]
pub(crate) struct WebgpuState {
    seed: u32,
    subtasks: HashMap<u32, SubtaskSlot>,
    /// Up to [`MAX_WAITABLES_PER_SET`] joined subtask handles per set.
    sets: HashMap<u32, Vec<u32>>,
}

impl WebgpuState {
    fn next_handle(&mut self) -> Result<u32, WebgpuErrno> {
        let id = self
            .seed
            .checked_add(1)
            .ok_or(WebgpuErrno::TooManyHandles)?;
        self.seed = id;
        Ok(id)
    }

    /// Registers a new pending subtask and returns its handle plus the
    /// `Notify` the caller's spawned background task must call
    /// `notify_one()` on once it resolves.
    pub fn insert_pending_subtask(&mut self) -> Result<(u32, Arc<Notify>), WebgpuErrno> {
        let handle = self.next_handle()?;
        let notify = Arc::new(Notify::new());
        self.subtasks
            .insert(handle, SubtaskSlot::Pending(notify.clone()));
        Ok((handle, notify))
    }

    /// Called by the spawned background task once its fake work completes.
    /// A no-op if the subtask was already dropped in the meantime.
    pub fn resolve_subtask(&mut self, handle: u32, payload: u32) {
        if self.subtasks.contains_key(&handle) {
            self.subtasks.insert(handle, SubtaskSlot::Resolved(payload));
        }
    }

    pub fn drop_subtask(&mut self, handle: u32) -> Result<(), WebgpuErrno> {
        self.subtasks
            .remove(&handle)
            .map(|_| ())
            .ok_or(WebgpuErrno::BadHandle)
    }

    pub fn new_waitable_set(&mut self) -> Result<u32, WebgpuErrno> {
        let handle = self.next_handle()?;
        self.sets.insert(handle, Vec::new());
        Ok(handle)
    }

    /// Joins `subtask` to `set`. Errors with [`WebgpuErrno::SetOccupied`]
    /// once `set` already holds [`MAX_WAITABLES_PER_SET`] members, rather
    /// than growing it unboundedly.
    pub fn join(&mut self, subtask: u32, set: u32) -> Result<(), WebgpuErrno> {
        if !self.subtasks.contains_key(&subtask) {
            return Err(WebgpuErrno::BadHandle);
        }
        let members = self.sets.get_mut(&set).ok_or(WebgpuErrno::BadHandle)?;
        if members.len() >= MAX_WAITABLES_PER_SET {
            return Err(WebgpuErrno::SetOccupied);
        }
        members.push(subtask);
        Ok(())
    }

    /// Non-blocking lookup used by `waitable_set_wait`'s fast path, and to
    /// obtain the `(handle, Notify)` pairs to await on the slow path.
    pub fn poll_wait(&self, set: u32) -> Result<WaitOutcome, WebgpuErrno> {
        let members = self.sets.get(&set).ok_or(WebgpuErrno::BadHandle)?;
        if members.is_empty() {
            return Err(WebgpuErrno::SetEmpty);
        }
        let mut pending = Vec::with_capacity(members.len());
        for &subtask in members {
            match self.subtasks.get(&subtask).ok_or(WebgpuErrno::BadHandle)? {
                SubtaskSlot::Resolved(payload) => return Ok(WaitOutcome::Ready(subtask, *payload)),
                SubtaskSlot::Pending(notify) => pending.push((subtask, notify.clone())),
            }
        }
        Ok(WaitOutcome::Pending(pending))
    }

    /// Removes `subtask` from `set`'s membership once its resolved event has
    /// been delivered to the guest via `waitable_set_wait` -- a subtask's
    /// terminal event fires once; after delivery it is no longer a
    /// candidate for future waits on this (or any) set. The subtask handle
    /// itself is untouched; the guest separately calls `subtask_drop` to
    /// free it. A no-op if `set` or `subtask` no longer exist.
    pub fn remove_from_set(&mut self, set: u32, subtask: u32) {
        if let Some(members) = self.sets.get_mut(&set) {
            members.retain(|&member| member != subtask);
        }
    }
}

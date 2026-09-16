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
//! wakeup, even if resolution races the check-and-drop-lock step. A waitable
//! set is `Option<u32>` -- at most one joined subtask handle, matching Phase
//! 1's single-waitable-per-set scope (see `syscalls/wasi_webgpu/mod.rs`'s
//! module doc comment for why that boundary is deliberate, not a bug).

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Notify;

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
    /// `waitable_join` targeted a set that already has a joined waitable --
    /// Phase 1 supports exactly one waitable per set.
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

/// Outcome of a non-blocking look-up of a waitable set's joined subtask.
pub(crate) enum WaitOutcome {
    /// Already resolved -- no need to await anything.
    Ready(u32),
    /// Still pending -- await this before re-checking.
    Pending(Arc<Notify>),
}

#[derive(Default, Debug)]
pub(crate) struct WebgpuState {
    seed: u32,
    subtasks: HashMap<u32, SubtaskSlot>,
    /// At most one joined subtask handle per set, per Phase 1's scope.
    sets: HashMap<u32, Option<u32>>,
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
        self.sets.insert(handle, None);
        Ok(handle)
    }

    /// Joins `subtask` to `set`. Errors with [`WebgpuErrno::SetOccupied`]
    /// rather than silently overwriting an existing join.
    pub fn join(&mut self, subtask: u32, set: u32) -> Result<(), WebgpuErrno> {
        if !self.subtasks.contains_key(&subtask) {
            return Err(WebgpuErrno::BadHandle);
        }
        let slot = self.sets.get_mut(&set).ok_or(WebgpuErrno::BadHandle)?;
        if slot.is_some() {
            return Err(WebgpuErrno::SetOccupied);
        }
        *slot = Some(subtask);
        Ok(())
    }

    /// Non-blocking lookup used by `waitable_set_wait`'s fast path, and to
    /// obtain the `Notify` to await on the slow path.
    pub fn poll_wait(&self, set: u32) -> Result<(u32, WaitOutcome), WebgpuErrno> {
        let subtask = self
            .sets
            .get(&set)
            .ok_or(WebgpuErrno::BadHandle)?
            .ok_or(WebgpuErrno::SetEmpty)?;
        match self.subtasks.get(&subtask).ok_or(WebgpuErrno::BadHandle)? {
            SubtaskSlot::Resolved(payload) => Ok((subtask, WaitOutcome::Ready(*payload))),
            SubtaskSlot::Pending(notify) => Ok((subtask, WaitOutcome::Pending(notify.clone()))),
        }
    }

    /// Re-checked after `.notified().await` resolves -- must be `Resolved`
    /// by then, since exactly one permit was reserved for this waiter.
    pub fn resolved_payload(&self, subtask: u32) -> Option<u32> {
        match self.subtasks.get(&subtask) {
            Some(SubtaskSlot::Resolved(payload)) => Some(*payload),
            _ => None,
        }
    }
}

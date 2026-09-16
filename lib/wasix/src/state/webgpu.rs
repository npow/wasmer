//! Handle tables for the `wasi_webgpu_v0` async spike -- see
//! `syscalls/wasi_webgpu/` for the host imports built on top of this.
//!
//! Lives on [`super::WasiState`] (the `Arc`-shared, per-process state), not
//! `WasiEnv` (cloned per thread) -- a subtask/future started by one thread
//! must be visible to and waitable-on by every other thread of the same
//! process, exactly like `WasiState::nn`.
//!
//! Handles do not survive `fork`, snapshot, or restore: each of those
//! rebuilds a fresh, empty [`WebgpuState`], exactly like `NnState`.
//!
//! Design: a subtask's lifecycle (started, then eventually resolved with a
//! payload) and a future's lifecycle (created empty, then eventually
//! written with a value) are the *same* representation: pending-with-a-
//! `Notify`, or resolved-with-a-`u32`. Rather than a second, near-identical
//! table, both are entries in one [`WaitableEntry`] table keyed by handle,
//! tagged with [`WaitableKind`] so kind-specific operations (`subtask_drop`
//! vs `future_drop`, `future_read`) reject a handle of the wrong kind
//! instead of silently accepting it. `join`/`poll_wait`/`remove_from_set`
//! and waitable sets themselves are already fully generic over "any
//! waitable with this lifecycle" and need no kind awareness at all -- a set
//! can mix subtasks and futures freely, matching the real Component Model's
//! own model of `waitable-set`s holding heterogeneous waitables.
//!
//! The spawning task (or, for a future, whatever eventually resolves it)
//! transitions the entry under this state's lock and calls
//! `Notify::notify_one()`; `Notify`'s single stored permit means a
//! `waitable_set_wait` caller that checks-then-awaits never misses the
//! wakeup, even if resolution races the check-and-drop-lock step.
//!
//! Phase 2a: a waitable set holds up to [`MAX_WAITABLES_PER_SET`] joined
//! waitable handles (`Vec<u32>`, was `Option<u32>` in Phase 1's
//! exactly-one-per-set scope). `waitable_set_wait` returns whichever member
//! resolves *first*, then removes it from the set -- a waitable's
//! resolution is a one-shot terminal event, so once delivered it is no
//! longer monitored; the guest separately calls `subtask_drop`/`future_drop`
//! to free the handle itself. The cap exists for the same reason
//! `examples/gpu_bridge.rs`'s `SessionLimits` does: a bounded per-instance
//! resource, not an unbounded one a hostile guest could grow without limit.
//!
//! `future.*`: unlike a subtask (always paired with a spawned background
//! task that resolves it), a bare `future_new` has no producer at all --
//! this bridge has no separate host/guest actors, so nothing plays the real
//! spec's `[future-writer]` role. `future_resolve_after` fills that gap: it
//! takes an *existing* future handle and spawns the same kind of background
//! task `request_adapter_start` spawns for its own subtask, just decoupled
//! from allocation. A guest-callable `future_write` was deliberately not
//! added -- this bridge's only realistic producer of a future's value is a
//! fake async host/GPU operation, never the guest itself, so a guest-facing
//! writer API would have no real caller.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Notify;

/// Bounded fan-in per waitable set (Phase 2a's scope: N joined waitables,
/// still no real multi-task reentrancy -- see `syscalls/wasi_webgpu/mod.rs`).
pub(crate) const MAX_WAITABLES_PER_SET: usize = 8;

/// Wire-level status codes for `wasi_webgpu_v0`'s host imports. Modeled
/// after `wasmer_wasi_nn::NnErrno`'s discriminant-per-condition style, but
/// local to this crate: this ABI has no separate published crate and never
/// will, since it is explicitly not a real `wasi:webgpu` implementation (see
/// `syscalls/wasi_webgpu/mod.rs`).
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
    /// The handle was never issued, was already released, or names a
    /// waitable of the wrong kind for this operation (e.g. `future_read` on
    /// a subtask handle).
    BadHandle = -4,
    /// `waitable_join` targeted a set already at [`MAX_WAITABLES_PER_SET`]
    /// members.
    SetOccupied = -5,
    /// `waitable_set_wait` targeted a set with no joined waitable.
    SetEmpty = -6,
    /// This instance's 32-bit handle counter is exhausted.
    TooManyHandles = -7,
    /// `future_read` on a future with no value written yet -- the guest
    /// must `waitable_join` it into a set and `waitable_set_wait`.
    Blocked = -8,
}

impl WebgpuErrno {
    pub fn to_i32(self) -> i32 {
        self as i32
    }
}

/// Which real Component Model resource type a [`WaitableEntry`] stands in
/// for. Purely a safety tag -- `join`/`poll_wait`/waitable sets never
/// inspect it, only kind-specific host imports do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WaitableKind {
    Subtask,
    Future,
}

#[derive(Debug)]
pub(crate) enum SlotState {
    Pending(Arc<Notify>),
    Resolved(u32),
}

#[derive(Debug)]
pub(crate) struct WaitableEntry {
    kind: WaitableKind,
    slot: SlotState,
}

/// Outcome of a non-blocking look-up of a waitable set's joined waitables.
pub(crate) enum WaitOutcome {
    /// One joined waitable has already resolved -- its handle and payload.
    /// No need to await anything.
    Ready(u32, u32),
    /// All joined waitables are still pending -- await any one of these
    /// (handle, `Notify`) pairs before re-polling. Never empty: only
    /// produced when the set has at least one member and none are resolved.
    Pending(Vec<(u32, Arc<Notify>)>),
}

#[derive(Default, Debug)]
pub(crate) struct WebgpuState {
    seed: u32,
    waitables: HashMap<u32, WaitableEntry>,
    /// Up to [`MAX_WAITABLES_PER_SET`] joined waitable handles per set.
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

    /// Registers a new pending waitable of the given kind and returns its
    /// handle. Used both by `request_adapter_start` (whose caller
    /// immediately spawns the resolving task itself) and by `future_new`
    /// (whose resolution, if any, comes later from `future_resolve_after`).
    /// Callers never need the `Notify` directly -- [`Self::resolve_waitable`]
    /// looks it up and fires it, so nothing can resolve a waitable while
    /// forgetting to wake its waiters.
    pub fn insert_pending(&mut self, kind: WaitableKind) -> Result<u32, WebgpuErrno> {
        let handle = self.next_handle()?;
        self.waitables.insert(
            handle,
            WaitableEntry {
                kind,
                slot: SlotState::Pending(Arc::new(Notify::new())),
            },
        );
        Ok(handle)
    }

    /// Called by whatever resolves a waitable (a spawned background task,
    /// or `future_resolve_after`'s) once it has a payload. Wakes any
    /// `waitable_set_wait` caller currently parked on this handle's
    /// `Notify` as part of the same transition -- resolving and waking are
    /// one atomic-under-the-lock step, not two the caller must remember to
    /// do in order. A no-op if the waitable was already dropped in the
    /// meantime.
    pub fn resolve_waitable(&mut self, handle: u32, payload: u32) {
        if let Some(entry) = self.waitables.get_mut(&handle) {
            if let SlotState::Pending(notify) = &entry.slot {
                notify.notify_one();
            }
            entry.slot = SlotState::Resolved(payload);
        }
    }

    /// Drops `handle`, but only if it exists and is of `expected_kind` --
    /// `subtask_drop` and `future_drop` each pass their own kind so neither
    /// can accidentally release (or be confused by) a handle of the other's
    /// kind.
    pub fn drop_waitable(
        &mut self,
        handle: u32,
        expected_kind: WaitableKind,
    ) -> Result<(), WebgpuErrno> {
        match self.waitables.get(&handle) {
            Some(entry) if entry.kind == expected_kind => {
                self.waitables.remove(&handle);
                Ok(())
            }
            _ => Err(WebgpuErrno::BadHandle),
        }
    }

    /// Non-blocking read of a future's value: `Ok(Some(value))` if resolved,
    /// `Ok(None)` if still pending (the guest must `waitable_join` +
    /// `waitable_set_wait`), or `Err(BadHandle)` if `handle` doesn't exist or
    /// isn't a future.
    pub fn read_future(&self, handle: u32) -> Result<Option<u32>, WebgpuErrno> {
        match self.waitables.get(&handle) {
            Some(entry) if entry.kind == WaitableKind::Future => match &entry.slot {
                SlotState::Resolved(payload) => Ok(Some(*payload)),
                SlotState::Pending(_) => Ok(None),
            },
            _ => Err(WebgpuErrno::BadHandle),
        }
    }

    pub fn new_waitable_set(&mut self) -> Result<u32, WebgpuErrno> {
        let handle = self.next_handle()?;
        self.sets.insert(handle, Vec::new());
        Ok(handle)
    }

    /// Joins `waitable` to `set`. Errors with [`WebgpuErrno::SetOccupied`]
    /// once `set` already holds [`MAX_WAITABLES_PER_SET`] members, rather
    /// than growing it unboundedly. `waitable` may be a subtask or a future
    /// -- a set does not care which.
    pub fn join(&mut self, waitable: u32, set: u32) -> Result<(), WebgpuErrno> {
        if !self.waitables.contains_key(&waitable) {
            return Err(WebgpuErrno::BadHandle);
        }
        let members = self.sets.get_mut(&set).ok_or(WebgpuErrno::BadHandle)?;
        if members.len() >= MAX_WAITABLES_PER_SET {
            return Err(WebgpuErrno::SetOccupied);
        }
        members.push(waitable);
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
        for &waitable in members {
            match &self
                .waitables
                .get(&waitable)
                .ok_or(WebgpuErrno::BadHandle)?
                .slot
            {
                SlotState::Resolved(payload) => return Ok(WaitOutcome::Ready(waitable, *payload)),
                SlotState::Pending(notify) => pending.push((waitable, notify.clone())),
            }
        }
        Ok(WaitOutcome::Pending(pending))
    }

    /// Removes `waitable` from `set`'s membership once its resolved event
    /// has been delivered to the guest via `waitable_set_wait` -- a
    /// waitable's terminal event fires once; after delivery it is no longer
    /// a candidate for future waits on this (or any) set. The waitable
    /// handle itself is untouched; the guest separately calls
    /// `subtask_drop`/`future_drop` to free it. A no-op if `set` or
    /// `waitable` no longer exist.
    pub fn remove_from_set(&mut self, set: u32, waitable: u32) {
        if let Some(members) = self.sets.get_mut(&set) {
            members.retain(|&member| member != waitable);
        }
    }
}

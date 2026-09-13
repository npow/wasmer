//! Handle tables for `wasi_ephemeral_nn` (`load` / `init_execution_context`).
//!
//! Lives on [`super::WasiState`] (the `Arc`-shared, per-process state), not on
//! `WasiEnv` (cloned per thread) -- a graph loaded by one thread must be visible
//! to every other thread of the same process, exactly like `WasiState::fs`.
//!
//! Handles do not survive `fork`, snapshot, or restore: each of those rebuilds
//! a fresh, empty [`NnState`].

use std::collections::HashMap;

use wasmer_wasi_nn::{NnErrno, NnExecutionContext, NnGraph};

#[derive(Default)]
pub(crate) struct NnState {
    seed: u32,
    graphs: HashMap<u32, Box<dyn NnGraph>>,
    contexts: HashMap<u32, Box<dyn NnExecutionContext>>,
}

impl std::fmt::Debug for NnState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnState")
            .field("graphs", &self.graphs.len())
            .field("contexts", &self.contexts.len())
            .finish()
    }
}

impl NnState {
    fn next_handle(&mut self) -> Result<u32, NnErrno> {
        let id = self.seed.checked_add(1).ok_or(NnErrno::TooLarge)?;
        self.seed = id;
        Ok(id)
    }

    pub fn insert_graph(&mut self, graph: Box<dyn NnGraph>) -> Result<u32, NnErrno> {
        let id = self.next_handle()?;
        self.graphs.insert(id, graph);
        Ok(id)
    }

    pub fn graph(&self, handle: u32) -> Option<&dyn NnGraph> {
        self.graphs.get(&handle).map(Box::as_ref)
    }

    pub fn insert_context(&mut self, ctx: Box<dyn NnExecutionContext>) -> Result<u32, NnErrno> {
        let id = self.next_handle()?;
        self.contexts.insert(id, ctx);
        Ok(id)
    }

    pub fn context_mut(&mut self, handle: u32) -> Option<&mut (dyn NnExecutionContext + 'static)> {
        self.contexts.get_mut(&handle).map(Box::as_mut)
    }
}

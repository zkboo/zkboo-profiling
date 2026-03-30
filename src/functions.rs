// SPDX-License-Identifier: LGPL-3.0-or-later

//! Implementation of profiling functions for ZKBoo circuits.

use crate::{ProfilingBackend, ProfilingData};
use zkboo::circuit::Circuit;

/// Produces profiling data based on all public circuit information.
///
/// Note: Information about word type and width is ingested as part of profiling,
///       but the specific values of input words are not, so that the circuit used for proof
///       generation and the circuit used for proof verification have the same profile.
pub fn profile<C: Circuit>(circuit: &C) -> ProfilingData {
    let mut profiler = ProfilingBackend::new().into_profiler();
    circuit.exec(&mut profiler);
    return profiler.finalize();
}

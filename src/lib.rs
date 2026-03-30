// SPDX-License-Identifier: LGPL-3.0-or-later

//! Profiling utilities for [zkboo] circuits.

#![no_std]
mod backend;
mod functions;

pub use backend::{GateCounts, ProfilingBackend, ProfilingData};
pub use functions::profile;

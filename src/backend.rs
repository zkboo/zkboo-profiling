// SPDX-License-Identifier: LGPL-3.0-or-later

//! Implementation of the profiling backend for ZKBoo circuits.

use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use zkboo::{
    backend::{Backend, Frontend},
    crypto::{Digest, Seed},
    memory::{FlexibleMemoryManager, MemoryManager, RefCount},
    word::{ByWordType, CompositeWord, Shape, Word, WordIdx},
};

/// Memory usage data, as a pair of number of bytes allocated on stack and on heap.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    pub stack: usize,
    pub heap: usize,
}

impl MemoryUsage {
    /// Total number of bytes allocated across stack and heap.
    pub fn total(&self) -> usize {
        return self.stack + self.heap;
    }

    /// Memory usage for a given words container.
    pub fn from_num_words(num_words: Shape) -> MemoryUsage {
        let usize_num_bytes = core::mem::size_of::<usize>();
        let mut usage = MemoryUsage { heap: 0, stack: 0 };
        let num_word_types = num_words.map(|_| 1).sum();
        usage.stack += 3 * usize_num_bytes * num_word_types;
        usage.heap += num_words.map_with_width(|w, n| w / 8 * n).sum();
        return usage;
    }
}

impl Add<Self> for MemoryUsage {
    type Output = Self;

    /// Adds two memory usages together, summing their stack and heap usage separately.
    fn add(self, other: Self) -> Self {
        return Self {
            stack: self.stack + other.stack,
            heap: self.heap + other.heap,
        };
    }
}

impl AddAssign<Self> for MemoryUsage {
    fn add_assign(&mut self, other: Self) {
        self.stack += other.stack;
        self.heap += other.heap;
    }
}

impl Sub<Self> for MemoryUsage {
    type Output = Self;

    /// Subtracts one memory usage from another, subtracting their stack and heap usage separately.
    fn sub(self, other: Self) -> Self {
        return Self {
            stack: self.stack - other.stack,
            heap: self.heap - other.heap,
        };
    }
}

impl SubAssign<Self> for MemoryUsage {
    fn sub_assign(&mut self, other: Self) {
        self.stack -= other.stack;
        self.heap -= other.heap;
    }
}

impl Mul<usize> for MemoryUsage {
    type Output = Self;

    /// Multiplies a memory usage by a scalar, multiplying its stack and heap usage separately.
    fn mul(self, rhs: usize) -> Self {
        return Self {
            stack: self.stack * rhs,
            heap: self.heap * rhs,
        };
    }
}

impl MulAssign<usize> for MemoryUsage {
    fn mul_assign(&mut self, rhs: usize) {
        self.stack *= rhs;
        self.heap *= rhs;
    }
}

/// Structure to track gate counts for all [Word] types.
/// [CompositeWord]s are counted according to their total number of machine [Word]s.
#[derive(Debug, Clone, Copy, Default)]
pub struct GateCounts {
    pub input: Shape,
    pub alloc: Shape,
    pub constant: Shape,
    pub from_le_words: Shape,
    pub to_le_words: Shape,
    pub output: Shape,
    pub not: Shape,
    pub bitxor: Shape,
    pub bitand: Shape,
    pub bitxor_const: Shape,
    pub bitand_const: Shape,
    pub unbounded_shl: Shape,
    pub unbounded_shr: Shape,
    pub rotate_left: Shape,
    pub rotate_right: Shape,
    pub reverse_bits: Shape,
    pub swap_bytes: Shape,
    pub cast: ByWordType<Shape>,
    pub carry: Shape,
}

/// Profiling data for a ZKBoo circuit.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProfilingData {
    gate_counts: GateCounts,
    state_size: Shape,
    max_live_wordrefs: Shape,
    max_cumulative_refcount: Shape,
    max_refcount: Shape,
}

/// Response data for a ZKBoo circuit, derived from its profiling data.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResponseData {
    and_msg_size: Shape,
    input_share_size: Shape,
}

impl ResponseData {
    /// Size of AND messages in the response, in bytes per [Word] type.
    pub fn and_msg_size(&self) -> Shape {
        return self.and_msg_size;
    }

    /// Size of input shares in the response, in bytes per [Word] type.
    pub fn input_share_size(&self) -> Shape {
        return self.input_share_size;
    }

    /// Memory usage for a ZKBoo response.
    pub fn mem_usage<D: Digest, S: Seed>(&self) -> MemoryUsage {
        let mut usage = MemoryUsage { heap: 0, stack: 0 };
        usage.stack += core::mem::size_of::<u8>(); // challenge
        usage.stack += 2 * core::mem::size_of::<S>(); // seeds
        usage.stack += core::mem::size_of::<D>(); // digest
        usage.heap += self.and_msg_size.map_with_width(|w, n| w / 8 * n).sum();
        usage.heap += self.input_share_size.map_with_width(|w, n| w / 8 * n).sum();
        return usage;
    }
}

/// Views data for a ZKBoo circuit, derived from its profiling data.
#[derive(Debug, Clone, Copy, Default)]
pub struct ViewsData {
    pub and_msgs_size: Shape,
    pub input_share2_size: Shape,
    pub output_shares_size: Shape,
}

impl ViewsData {
    /// Size of AND messages in the views, in bytes per [Word] type.
    pub fn and_msgs_size(&self) -> Shape {
        return self.and_msgs_size;
    }

    /// Size of input shares for party 2 in the views, in bytes per [Word] type.
    pub fn input_share2_size(&self) -> Shape {
        return self.input_share2_size;
    }

    /// Size of output shares in the views, in bytes per [Word] type.
    pub fn output_shares_size(&self) -> Shape {
        return self.output_shares_size;
    }
}

impl ProfilingData {
    /// Size of the circuit state, in bytes per [Word] type.
    pub fn state_size(&self) -> Shape {
        return self.state_size;
    }

    /// Counts of each gate type in the circuit, in number of gates per [Word] type.
    pub fn gate_counts(&self) -> &GateCounts {
        return &self.gate_counts;
    }

    /// Maximum number of live word references at any point during execution,
    /// in number of word references per [Word] type.
    pub fn max_live_wordrefs(&self) -> Shape {
        return self.max_live_wordrefs;
    }

    /// Maximum cumulative reference count at any point during execution,
    /// in number of word references per [Word] type.
    pub fn max_cumulative_refcount(&self) -> Shape {
        return self.max_cumulative_refcount;
    }

    /// Maximum reference count for any individual word reference at any point during execution,
    pub fn max_refcount(&self) -> Shape {
        return self.max_refcount;
    }

    /// Size of AND messages in the response, in bytes per [Word] type.
    pub fn and_msg_size(&self) -> Shape {
        return self
            .gate_counts
            .bitand
            .zip(&self.gate_counts.carry, |bitand_count, carry_count| {
                bitand_count + carry_count
            });
    }

    /// Derives response data from this profiling data.
    pub fn response_data(&self) -> ResponseData {
        return ResponseData {
            and_msg_size: self.and_msg_size(),
            input_share_size: self.gate_counts.input,
        };
    }

    /// Derives views data from this profiling data.
    pub fn views_data(&self) -> ViewsData {
        return ViewsData {
            and_msgs_size: self.and_msg_size().map(|n| n * 3),
            input_share2_size: self.gate_counts.input,
            output_shares_size: self.gate_counts.output.map(|n| n * 3),
        };
    }

    /// Memory usage for the circuit state, based on the maximum state size during execution.
    pub fn state_mem_usage(&self) -> MemoryUsage {
        return MemoryUsage::from_num_words(self.state_size);
    }

    /// Memory usage for the circuit output, based on the total number of output words.
    pub fn output_mem_usage(&self) -> MemoryUsage {
        return MemoryUsage::from_num_words(self.gate_counts.output);
    }

    /// Memory usage for word references, based on the maximum number of live word references
    /// and maximum cumulative reference count during execution.
    pub fn wordrefs_mem_usage(&self) -> MemoryUsage {
        let usize_num_bytes = core::mem::size_of::<usize>();
        let mut usage = MemoryUsage { heap: 0, stack: 0 };
        let num_wordrefs = self.max_live_wordrefs.sum();
        let num_idxs = self.max_cumulative_refcount.sum();
        usage.stack += usize_num_bytes * num_wordrefs; // Rc pointer
        usage.stack += usize_num_bytes * num_idxs; // idxs
        return usage;
    }

    /// Memory usage for the memory manager, based on the maximum state size during execution and
    /// the number of reference count updates.
    pub fn memory_manager_mem_usage<RC: RefCount>(&self) -> MemoryUsage {
        let usize_num_bytes = core::mem::size_of::<usize>();
        let rc_num_bytes = core::mem::size_of::<RC>();
        let mut usage = MemoryUsage { heap: 0, stack: 0 };
        let state_size = self.state_size;
        let num_word_types = state_size.map(|_| 1).sum();
        usage.stack += 3 * usize_num_bytes * num_word_types; // Vec<RC>
        usage.stack += 6 * usize_num_bytes * num_word_types; // AllocSet
        usage.heap += rc_num_bytes * state_size.sum(); // Vec<RC>
        usage.heap += rc_num_bytes * state_size.map(|v| (v + 63) / 64).sum(); // AllocSet
        return usage;
    }

    /// Total memory usage for the executor, based on the memory usage of the circuit state,
    /// word references, memory manager, and output.
    pub fn executor_mem_usage<RC: RefCount>(&self) -> MemoryUsage {
        let mut usage = MemoryUsage { heap: 0, stack: 0 };
        usage += self.state_mem_usage();
        usage += self.wordrefs_mem_usage();
        usage += self.memory_manager_mem_usage::<RC>();
        usage += self.output_mem_usage();
        return usage;
    }

    /// Memory usage for the prover, based on the memory usage of the circuit state shares,
    /// word references, and memory manager.
    pub fn prover_mem_usage<RC: RefCount>(&self) -> MemoryUsage {
        let mut usage = MemoryUsage { heap: 0, stack: 0 };
        usage += self.state_mem_usage() * 3;
        usage += self.wordrefs_mem_usage();
        usage += self.memory_manager_mem_usage::<RC>();
        return usage;
    }

    /// Memory usage for the verifier, based on the memory usage of the circuit state shares,
    /// word references, and memory manager.
    pub fn verifier_mem_usage<RC: RefCount>(&self) -> MemoryUsage {
        let mut usage = MemoryUsage { heap: 0, stack: 0 };
        usage += self.state_mem_usage() * 2;
        usage += self.wordrefs_mem_usage();
        usage += self.memory_manager_mem_usage::<RC>();
        return usage;
    }
}

/// Profiling backend for ZKBoo circuits.
#[derive(Debug)]
pub struct ProfilingBackend {
    data: ProfilingData,
    memory_manager: FlexibleMemoryManager<usize>,
    live_wordrefs: Shape,
    cumulative_refcount: Shape,
}

impl ProfilingBackend {
    /// Create a new profiling backend.
    pub fn new() -> Self {
        return Self {
            data: ProfilingData::default(),
            memory_manager: FlexibleMemoryManager::new(),
            live_wordrefs: Shape::zero(),
            cumulative_refcount: Shape::zero(),
        };
    }

    /// Wraps this profiling backend into a [Frontend].
    ///
    /// Alias of [Backend::into_frontend].
    pub fn into_profiler(self) -> Frontend<Self> {
        return self.into_frontend();
    }
}

impl Backend for ProfilingBackend {
    type FinalizeArg = ();
    type FinalizeResult = ProfilingData;

    fn finalize(self, _arg: Self::FinalizeArg) -> Self::FinalizeResult {
        return self.data;
    }

    fn input<W: Word, const N: usize>(&mut self, _word: CompositeWord<W, N>) -> WordIdx<W, N> {
        let (idx, size) = self.memory_manager.alloc::<W, N>();
        *self.data.gate_counts.input.as_value_mut::<W>() += N;
        *self.data.state_size.as_value_mut::<W>() = size;
        return idx;
    }

    fn alloc<W: Word, const N: usize>(&mut self) -> WordIdx<W, N> {
        let (idx, size) = self.memory_manager.alloc::<W, N>();
        *self.data.gate_counts.alloc.as_value_mut::<W>() += N;
        *self.data.state_size.as_value_mut::<W>() = size;
        return idx;
    }

    fn constant<W: Word, const N: usize>(
        &mut self,
        _word: CompositeWord<W, N>,
        _out: WordIdx<W, N>,
    ) {
        *self.data.gate_counts.constant.as_value_mut::<W>() += N;
    }

    fn from_le_words<W: Word, const N: usize>(
        &mut self,
        _ins: [WordIdx<W, 1>; N],
        _out: WordIdx<W, N>,
    ) {
        *self.data.gate_counts.from_le_words.as_value_mut::<W>() += N;
    }

    fn to_le_words<W: Word, const N: usize>(
        &mut self,
        _in_: WordIdx<W, N>,
        _outs: [WordIdx<W, 1>; N],
    ) {
        *self.data.gate_counts.to_le_words.as_value_mut::<W>() += N;
    }

    fn output<W: Word, const N: usize>(&mut self, _out: WordIdx<W, N>) {
        *self.data.gate_counts.output.as_value_mut::<W>() += N;
    }

    fn increase_refcount<W: Word, const N: usize>(&mut self, idx: WordIdx<W, N>) {
        self.memory_manager.increase_refcount(idx);
        // Update max cumulative refcount:
        let cumulative_refcount = self.cumulative_refcount.as_value_mut::<W>();
        let max_cumulative_refcount = self.data.max_cumulative_refcount.as_value_mut::<W>();
        *cumulative_refcount += N;
        if cumulative_refcount > max_cumulative_refcount {
            *max_cumulative_refcount = *cumulative_refcount;
        }
        // Update max live wordrefs:
        let live_wordrefs = self.live_wordrefs.as_value_mut::<W>();
        let max_live_wordrefs = self.data.max_live_wordrefs.as_value_mut::<W>();
        *live_wordrefs += 1;
        if live_wordrefs > max_live_wordrefs {
            *max_live_wordrefs = *live_wordrefs;
        }
        // Update max refcount:
        let refcount = self.memory_manager.refcounts().as_vec::<W>()[idx.into_array()[0]];
        let max_refcount = self.data.max_refcount.as_value_mut::<W>();
        if refcount > *max_refcount {
            *max_refcount = refcount;
        }
    }

    fn decrease_refcount<W: Word, const N: usize>(&mut self, idx: WordIdx<W, N>) {
        self.memory_manager.decrease_refcount(idx);
        *self.cumulative_refcount.as_value_mut::<W>() -= N;
        *self.live_wordrefs.as_value_mut::<W>() -= 1;
    }

    fn not<W: Word, const N: usize>(&mut self, _in_: WordIdx<W, N>, _out: WordIdx<W, N>) {
        *self.data.gate_counts.not.as_value_mut::<W>() += N;
    }

    fn bitxor<W: Word, const N: usize>(
        &mut self,
        _inl: WordIdx<W, N>,
        _inr: WordIdx<W, N>,
        _out: WordIdx<W, N>,
    ) {
        *self.data.gate_counts.bitxor.as_value_mut::<W>() += N;
    }

    fn bitand<W: Word, const N: usize>(
        &mut self,
        _inl: WordIdx<W, N>,
        _inr: WordIdx<W, N>,
        _out: WordIdx<W, N>,
    ) {
        *self.data.gate_counts.bitand.as_value_mut::<W>() += N;
    }

    fn bitxor_const<W: Word, const N: usize>(
        &mut self,
        _inl: WordIdx<W, N>,
        _inr: CompositeWord<W, N>,
        _out: WordIdx<W, N>,
    ) {
        *self.data.gate_counts.bitxor_const.as_value_mut::<W>() += N;
    }

    fn bitand_const<W: Word, const N: usize>(
        &mut self,
        _inl: WordIdx<W, N>,
        _inr: CompositeWord<W, N>,
        _out: WordIdx<W, N>,
    ) {
        *self.data.gate_counts.bitand_const.as_value_mut::<W>() += N;
    }

    fn unbounded_shl<W: Word, const N: usize>(
        &mut self,
        _in_: WordIdx<W, N>,
        _shift: usize,
        _out: WordIdx<W, N>,
    ) {
        if N == 1 {
            *self.data.gate_counts.unbounded_shl.as_value_mut::<W>() += 1;
        } else {
            *self.data.gate_counts.rotate_left.as_value_mut::<W>() += N;
            *self.data.gate_counts.bitand_const.as_value_mut::<W>() += 2 * N;
            *self.data.gate_counts.bitxor.as_value_mut::<W>() += N;
        }
    }

    fn unbounded_shr<W: Word, const N: usize>(
        &mut self,
        _in_: WordIdx<W, N>,
        _shift: usize,
        _out: WordIdx<W, N>,
    ) {
        if N == 1 {
            *self.data.gate_counts.unbounded_shr.as_value_mut::<W>() += 1;
        } else {
            *self.data.gate_counts.rotate_right.as_value_mut::<W>() += N;
            *self.data.gate_counts.bitand_const.as_value_mut::<W>() += 2 * N;
            *self.data.gate_counts.bitxor.as_value_mut::<W>() += N;
        }
    }

    fn rotate_left<W: Word, const N: usize>(
        &mut self,
        _in_: WordIdx<W, N>,
        _shift: usize,
        _out: WordIdx<W, N>,
    ) {
        if N == 1 {
            *self.data.gate_counts.rotate_left.as_value_mut::<W>() += 1;
        } else {
            *self.data.gate_counts.rotate_left.as_value_mut::<W>() += N;
            *self.data.gate_counts.bitand_const.as_value_mut::<W>() += 2 * N;
            *self.data.gate_counts.bitxor.as_value_mut::<W>() += N;
        }
    }

    fn rotate_right<W: Word, const N: usize>(
        &mut self,
        _in_: WordIdx<W, N>,
        _shift: usize,
        _out: WordIdx<W, N>,
    ) {
        if N == 1 {
            *self.data.gate_counts.rotate_right.as_value_mut::<W>() += 1;
        } else {
            *self.data.gate_counts.rotate_right.as_value_mut::<W>() += N;
            *self.data.gate_counts.bitand_const.as_value_mut::<W>() += 2 * N;
            *self.data.gate_counts.bitxor.as_value_mut::<W>() += N;
        }
    }

    fn reverse_bits<W: Word, const N: usize>(&mut self, _in_: WordIdx<W, N>, _out: WordIdx<W, N>) {
        *self.data.gate_counts.reverse_bits.as_value_mut::<W>() += N;
    }

    fn swap_bytes<W: Word, const N: usize>(&mut self, _in_: WordIdx<W, N>, _out: WordIdx<W, N>) {
        *self.data.gate_counts.swap_bytes.as_value_mut::<W>() += N;
    }

    fn cast<W: Word, T: Word>(&mut self, _in_: WordIdx<W, 1>, _out: WordIdx<T, 1>) {
        *self
            .data
            .gate_counts
            .cast
            .as_value_mut::<W>()
            .as_value_mut::<T>() += 1;
    }

    fn carry<W: Word, const N: usize>(
        &mut self,
        _p: WordIdx<W, N>,
        _g: WordIdx<W, N>,
        _carry_in: bool,
        _out: WordIdx<W, N>,
    ) {
        *self.data.gate_counts.carry.as_value_mut::<W>() += N;
    }
}

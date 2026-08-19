// SPDX-License-Identifier: LGPL-3.0-or-later

//! Gate-count pin for the in-circuit [`WordRef::wide_mul`].
//!
//! `wide_mul` uses a right-shifting carry-save accumulator that emits exactly two AND messages per
//! row (one for the partial product, one for the carry-save majority) plus one final resolving
//! addition. For a 256-bit product (`CompositeWord<u64, 4>`) that is `2 * 256 * 4 = 2048` word-level
//! AND messages from the rows, plus `8` from the closing add, for `2056` in total — roughly half the
//! `4123` of the earlier double-width, left-shifting formulation. This test pins that figure so a
//! regression that reintroduces the extra AND per row is caught immediately.

#[cfg(feature = "u64")]
use zkboo::{
    backend::{Backend, Frontend},
    circuit::Circuit,
    word::CompositeWord,
};
#[cfg(feature = "u64")]
use zkboo_profiling::profile;

#[cfg(feature = "u64")]
struct WideMul256 {
    a: CompositeWord<u64, 4>,
    b: CompositeWord<u64, 4>,
}

#[cfg(feature = "u64")]
impl Circuit for WideMul256 {
    fn exec<B: Backend>(&self, frontend: &Frontend<B>) {
        let a = frontend.input(self.a);
        let b = frontend.input(self.b);
        let (lo, hi) = a.wide_mul(b);
        frontend.output(lo);
        frontend.output(hi);
    }
}

#[cfg(feature = "u64")]
#[test]
fn test_wide_mul_256_and_msg_size() {
    let circuit = WideMul256 {
        a: CompositeWord::<u64, 4>::MAX,
        b: CompositeWord::<u64, 4>::MAX,
    };
    let and_msg_size = profile(&circuit).and_msg_size().sum();
    assert_eq!(
        and_msg_size, 2056,
        "wide_mul AND-message count changed (expected 2 ANDs/row + final add for a 256-bit product)"
    );
}

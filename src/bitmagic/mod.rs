use core::iter::zip;
use equator::assert;
use faer::MatRef;

pub mod matmul;
pub mod matvec;
pub mod tmatvec;

pub fn matvec_bit(m: usize, n: usize, dst: &mut [f32], lhs: MatRef<'_, f32>, rhs: &[u16]) {
    struct Impl<'a> {
        m: usize,
        n: usize,
        dst: &'a mut [f32],
        lhs: MatRef<'a, f32>,
        rhs: &'a [u16],
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                m,
                n,
                dst,
                lhs,
                rhs,
            } = self;

            assert!(all(
                lhs.nrows() == m,
                lhs.ncols() == n,
                rhs.len() == n.div_ceil(16),
                dst.len() == m,
            ));

            for j in 0..n {
                let lhs = lhs.col(j).try_as_slice().unwrap();
                let pos = (rhs[j / 16] >> (j % 16)) & 1 == 0;
                if pos {
                    for (dst, &lhs) in zip(&mut *dst, lhs) {
                        *dst += lhs;
                    }
                } else {
                    for (dst, &lhs) in zip(&mut *dst, lhs) {
                        *dst -= lhs;
                    }
                }
            }
        }
    }

    pulp::Arch::new().dispatch(Impl {
        m,
        n,
        dst,
        lhs,
        rhs,
    })
}

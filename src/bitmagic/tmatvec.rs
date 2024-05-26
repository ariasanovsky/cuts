use crate::SignMatRef;
use equator::assert;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86 {
    use super::*;
    use core::iter::zip;
    use pulp::{cast, x86::V3, Simd};

    pub fn tmatvec_f32_v4(simd: pulp::x86::V4, dst: &mut [f32], lhs: SignMatRef<'_>, rhs: &[f32]) {
        struct Impl<'a> {
            simd: pulp::x86::V4,
            dst: &'a mut [f32],
            lhs: SignMatRef<'a>,
            rhs: &'a [f32],
        }

        impl pulp::NullaryFnOnce for Impl<'_> {
            type Output = ();

            #[inline(always)]
            fn call(self) -> Self::Output {
                let Self {
                    simd,
                    dst,
                    lhs,
                    rhs,
                } = self;

                use pulp::{b16, f32x16};

                let (rhs_head, rhs_tail) = pulp::as_arrays::<16, _>(rhs);
                let rhs_head: &[f32x16] = bytemuck::cast_slice(rhs_head);

                let m = lhs.nrows();
                let n = lhs.ncols();

                let pos_one = simd.splat_f32x16(1.0);
                let neg_one = simd.splat_f32x16(-1.0);

                for j in 0..n {
                    let lhs = bytemuck::cast_slice::<_, u16>(lhs.storage().col_as_slice(j));

                    let mut acc0 = simd.splat_f32x16(0.0);
                    let mut acc1 = simd.splat_f32x16(0.0);
                    let mut acc2 = simd.splat_f32x16(0.0);
                    let mut acc3 = simd.splat_f32x16(0.0);

                    {
                        let lhs = &lhs[..m / 16];

                        let (lhs_head, lhs_tail) = pulp::as_arrays::<4, _>(lhs);
                        let (rhs_head, rhs_tail) = pulp::as_arrays::<4, _>(rhs_head);

                        for (lhs, rhs) in zip(lhs_head, rhs_head) {
                            {
                                let lhs = lhs[0];
                                let rhs = rhs[0];
                                acc0 = simd.mul_add_f32x16(
                                    simd.select_f32x16(b16(lhs), neg_one, pos_one),
                                    rhs,
                                    acc0,
                                );
                            }
                            {
                                let lhs = lhs[1];
                                let rhs = rhs[1];
                                acc1 = simd.mul_add_f32x16(
                                    simd.select_f32x16(b16(lhs), neg_one, pos_one),
                                    rhs,
                                    acc1,
                                );
                            }
                            {
                                let lhs = lhs[2];
                                let rhs = rhs[2];
                                acc2 = simd.mul_add_f32x16(
                                    simd.select_f32x16(b16(lhs), neg_one, pos_one),
                                    rhs,
                                    acc2,
                                );
                            }
                            {
                                let lhs = lhs[3];
                                let rhs = rhs[3];
                                acc3 = simd.mul_add_f32x16(
                                    simd.select_f32x16(b16(lhs), neg_one, pos_one),
                                    rhs,
                                    acc3,
                                );
                            }
                        }

                        for (lhs, rhs) in zip(lhs_tail, rhs_tail) {
                            {
                                let lhs = *lhs;
                                let rhs = *rhs;
                                acc0 = simd.mul_add_f32x16(
                                    simd.select_f32x16(b16(lhs), neg_one, pos_one),
                                    rhs,
                                    acc0,
                                );
                            }
                        }
                    }

                    if m % 16 != 0 {
                        let lhs = &lhs[m / 16];

                        let rhs = simd.f32s_partial_load(&rhs_tail[..]);
                        acc0 = simd.mul_add_f32x16(
                            simd.select_f32x16(b16(*lhs), neg_one, pos_one),
                            rhs,
                            acc0,
                        );
                    }

                    acc0 = simd.add_f32x16(acc0, acc1);
                    acc2 = simd.add_f32x16(acc2, acc3);
                    acc0 = simd.add_f32x16(acc0, acc2);

                    dst[j] += simd.f32s_reduce_sum(acc0);
                }
            }
        }

        simd.vectorize(Impl {
            simd,
            dst,
            lhs,
            rhs,
        });
    }

    pub fn tmatvec_f32_v3(simd: V3, dst: &mut [f32], lhs: SignMatRef<'_>, rhs: &[f32]) {
        struct Impl<'a> {
            simd: V3,
            dst: &'a mut [f32],
            lhs: SignMatRef<'a>,
            rhs: &'a [f32],
        }

        impl pulp::NullaryFnOnce for Impl<'_> {
            type Output = ();

            #[inline(always)]
            fn call(self) -> Self::Output {
                let Self {
                    simd,
                    dst,
                    lhs,
                    rhs,
                } = self;

                use pulp::{f32x8, u32x8};

                let (rhs_head, rhs_tail) = pulp::as_arrays::<8, _>(rhs);
                let rhs_head: &[f32x8] = bytemuck::cast_slice(rhs_head);
                let (rhs_head4, rhs_head1) = pulp::as_arrays::<4, _>(rhs_head);

                let m = lhs.nrows();
                let n = lhs.ncols();

                const MASK: u32x8 = u32x8(
                    (1 << 0) | (1 << 8) | (1 << 16) | (1 << 24),
                    (1 << 1) | (1 << 9) | (1 << 17) | (1 << 25),
                    (1 << 2) | (1 << 10) | (1 << 18) | (1 << 26),
                    (1 << 3) | (1 << 11) | (1 << 19) | (1 << 27),
                    (1 << 4) | (1 << 12) | (1 << 20) | (1 << 28),
                    (1 << 5) | (1 << 13) | (1 << 21) | (1 << 29),
                    (1 << 6) | (1 << 14) | (1 << 22) | (1 << 30),
                    (1 << 7) | (1 << 15) | (1 << 23) | (1 << 31),
                );
                const SHIFT: u32x8 = u32x8(7, 6, 5, 4, 3, 2, 1, 0);
                let sign_bit = simd.splat_u32x8(u32::MAX / 2 + 1);

                for j in 0..n {
                    let lhs = bytemuck::cast_slice::<_, u8>(lhs.storage().col_as_slice(j));

                    let mut acc0 = simd.splat_f32x8(0.0);
                    let mut acc1 = simd.splat_f32x8(0.0);
                    let mut acc2 = simd.splat_f32x8(0.0);
                    let mut acc3 = simd.splat_f32x8(0.0);

                    {
                        let lhs = &lhs[..m / 8];
                        let (lhs_head4, lhs_head1) = pulp::as_arrays::<4, _>(lhs);

                        let lhs_head = bytemuck::cast_slice::<_, u32>(lhs_head4);

                        for (lhs, [rhs0, rhs1, rhs2, rhs3]) in zip(lhs_head, &*rhs_head4) {
                            let lhs = simd
                                .shl_dyn_u32x8(simd.and_u32x8(simd.splat_u32x8(*lhs), MASK), SHIFT);
                            {
                                let lhs = simd.shl_const_u32x8::<24>(lhs);
                                let acc = simd.xor_f32x8(cast(lhs), *rhs0);
                                acc0 = simd.add_f32x8(acc0, acc);
                            }
                            {
                                let lhs = simd.shl_const_u32x8::<16>(lhs);
                                let acc =
                                    simd.xor_f32x8(cast(simd.and_u32x8(sign_bit, lhs)), *rhs1);
                                acc1 = simd.add_f32x8(acc1, acc);
                            }
                            {
                                let lhs = simd.shl_const_u32x8::<8>(lhs);
                                let acc =
                                    simd.xor_f32x8(cast(simd.and_u32x8(sign_bit, lhs)), *rhs2);
                                acc2 = simd.add_f32x8(acc2, acc);
                            }
                            {
                                let lhs = simd.shl_const_u32x8::<0>(lhs);
                                let acc =
                                    simd.xor_f32x8(cast(simd.and_u32x8(sign_bit, lhs)), *rhs3);
                                acc3 = simd.add_f32x8(acc3, acc);
                            }
                        }
                        for (lhs, rhs0) in zip(lhs_head1, rhs_head1) {
                            let lhs = simd.shl_dyn_u32x8(
                                simd.and_u32x8(simd.splat_u32x8(*lhs as u32), MASK),
                                SHIFT,
                            );
                            {
                                let lhs = simd.shl_const_u32x8::<24>(lhs);
                                let acc = simd.xor_f32x8(cast(lhs), *rhs0);
                                acc0 = simd.add_f32x8(acc0, acc);
                            }
                        }
                    }

                    if m % 8 != 0 {
                        let lhs = &lhs[m / 8];

                        let rhs = simd.f32s_partial_load(rhs_tail);
                        let lhs = simd.shl_const_u32x8::<24>(simd.shl_dyn_u32x8(
                            simd.and_u32x8(simd.splat_u32x8(*lhs as u32), MASK),
                            SHIFT,
                        ));
                        let acc = simd.xor_f32x8(cast(lhs), rhs);
                        acc0 = simd.add_f32x8(acc0, acc);
                    }

                    acc0 = simd.add_f32x8(acc0, acc1);
                    acc2 = simd.add_f32x8(acc2, acc3);
                    acc0 = simd.add_f32x8(acc0, acc2);

                    dst[j] += simd.f32s_reduce_sum(acc0);
                }
            }
        }

        simd.vectorize(Impl {
            simd,
            dst,
            lhs,
            rhs,
        });
    }
}

pub fn tmatvec_f32_scalar(dst: &mut [f32], lhs: SignMatRef<'_>, rhs: &[f32]) {
    let m = lhs.nrows();
    let n = lhs.ncols();

    for j in 0..n {
        let lhs = bytemuck::cast_slice::<_, u8>(lhs.storage().col_as_slice(j));
        let mut acc = 0.0;

        for i in 0..m {
            let div = i / 8;
            let rem = i % 8;

            let lhs = (lhs[div] >> rem) & 1 == 1;
            let rhs = rhs[i];

            acc += if lhs { -rhs } else { rhs };
        }
        dst[j] += acc;
    }
}

pub fn tmatvec_f32(dst: &mut [f32], lhs: SignMatRef<'_>, rhs: &[f32]) {
    assert!(all(lhs.nrows() == rhs.len(), lhs.ncols() == dst.len()));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(simd) = pulp::x86::V4::try_new() {
            return x86::tmatvec_f32_v4(simd, dst, lhs, rhs);
        }
        if let Some(simd) = pulp::x86::V3::try_new() {
            return x86::tmatvec_f32_v3(simd, dst, lhs, rhs);
        }
    }

    tmatvec_f32_scalar(dst, lhs, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MatRef;
    use equator::assert;
    use rand::prelude::*;

    #[test]
    fn test_v4() {
        if let Some(simd) = pulp::x86::V4::try_new() {
            let m = 421usize;
            let n = 30usize;

            let rng = &mut StdRng::seed_from_u64(0);

            let stride = m.div_ceil(64);
            let data = (0..n * stride)
                .map(|_| rng.gen::<u64>())
                .collect::<Vec<_>>();

            let lhs =
                SignMatRef::from_storage(MatRef::from_col_major_slice(&data, stride, n, stride), m);

            let rhs = (0..m).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();
            let acc = (0..n).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();

            let mut dst_actual = acc.clone();
            let mut dst_target = acc.clone();

            tmatvec_f32_scalar(&mut dst_target, lhs, &rhs);
            x86::tmatvec_f32_v4(simd, &mut dst_actual, lhs, &rhs);

            for (actual, target) in core::iter::zip(&dst_actual, &dst_target) {
                assert!((actual - target).abs() <= 1e-4);
            }
        }
    }

    #[test]
    fn test_v3() {
        if let Some(simd) = pulp::x86::V3::try_new() {
            let m = 421usize;
            let n = 30usize;

            let rng = &mut StdRng::seed_from_u64(0);

            let stride = m.div_ceil(64);
            let data = (0..n * stride)
                .map(|_| rng.gen::<u64>())
                .collect::<Vec<_>>();

            let lhs =
                SignMatRef::from_storage(MatRef::from_col_major_slice(&data, stride, n, stride), m);

            let rhs = (0..m).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();
            let acc = (0..n).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();

            let mut dst_actual = acc.clone();
            let mut dst_target = acc.clone();

            tmatvec_f32_scalar(&mut dst_target, lhs, &rhs);
            x86::tmatvec_f32_v3(simd, &mut dst_actual, lhs, &rhs);

            for (actual, target) in core::iter::zip(&dst_actual, &dst_target) {
                assert!((actual - target).abs() <= 1e-4);
            }
        }
    }
}

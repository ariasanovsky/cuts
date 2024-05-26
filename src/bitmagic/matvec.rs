use crate::SignMatRef;
use equator::assert;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86 {
    use super::*;
    use core::iter::zip;
    use pulp::{cast, x86::V3, Simd};

    pub fn matvec_f32_v4(simd: pulp::x86::V4, dst: &mut [f32], lhs: SignMatRef<'_>, rhs: &[f32]) {
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

                let (dst_head, dst_tail) = pulp::as_arrays_mut::<16, _>(dst);
                let dst_head: &mut [f32x16] = bytemuck::cast_slice_mut(dst_head);

                let m = lhs.nrows();
                let n = lhs.ncols();

                for j in 0..n {
                    let lhs = bytemuck::cast_slice::<_, u16>(lhs.storage().col_as_slice(j));

                    let rhs = rhs[j];
                    let pos_rhs = simd.splat_f32x16(rhs);
                    let neg_rhs = simd.splat_f32x16(-rhs);

                    {
                        let lhs = &lhs[..m / 16];

                        for (dst, lhs) in zip(&mut *dst_head, lhs) {
                            let mut dst_ = *dst;
                            let acc = simd.select_f32x16(b16(*lhs), neg_rhs, pos_rhs);
                            dst_ = simd.add_f32x16(dst_, acc);
                            *dst = dst_;
                        }
                    }

                    if m % 16 != 0 {
                        let lhs = &lhs[m / 16];

                        let mut dst_ = simd.f32s_partial_load(dst_tail);
                        let acc = simd.select_f32x16(b16(*lhs), neg_rhs, pos_rhs);
                        dst_ = simd.add_f32x16(dst_, acc);
                        simd.f32s_partial_store(dst_tail, dst_);
                    }
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

    pub fn matvec_f32_v3(simd: V3, dst: &mut [f32], lhs: SignMatRef<'_>, rhs: &[f32]) {
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

                let (dst_head, dst_tail) = pulp::as_arrays_mut::<8, _>(dst);
                let dst_head: &mut [f32x8] = bytemuck::cast_slice_mut(dst_head);

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

                    let rhs = rhs[j];
                    let rhs = simd.splat_f32x8(rhs);

                    {
                        let lhs = &lhs[..m / 8];
                        let (lhs_head, lhs_tail) = pulp::as_arrays::<4, _>(lhs);
                        let (dst_head, dst_tail) = pulp::as_arrays_mut::<4, _>(dst_head);

                        let lhs_head = bytemuck::cast_slice::<_, u32>(lhs_head);

                        for ([dst0, dst1, dst2, dst3], lhs) in zip(&mut *dst_head, lhs_head) {
                            let lhs = simd
                                .shl_dyn_u32x8(simd.and_u32x8(simd.splat_u32x8(*lhs), MASK), SHIFT);
                            {
                                let lhs = simd.shl_const_u32x8::<24>(lhs);
                                let acc = simd.xor_f32x8(cast(lhs), rhs);
                                *dst0 = simd.add_f32x8(*dst0, acc);
                            }
                            {
                                let lhs = simd.shl_const_u32x8::<16>(lhs);
                                let acc = simd.xor_f32x8(cast(simd.and_u32x8(sign_bit, lhs)), rhs);
                                *dst1 = simd.add_f32x8(*dst1, acc);
                            }
                            {
                                let lhs = simd.shl_const_u32x8::<8>(lhs);
                                let acc = simd.xor_f32x8(cast(simd.and_u32x8(sign_bit, lhs)), rhs);
                                *dst2 = simd.add_f32x8(*dst2, acc);
                            }
                            {
                                let acc = simd.xor_f32x8(cast(simd.and_u32x8(sign_bit, lhs)), rhs);
                                *dst3 = simd.add_f32x8(*dst3, acc);
                            }
                        }
                        for (dst, lhs) in zip(&mut *dst_tail, lhs_tail) {
                            let mut dst_ = *dst;
                            let lhs = simd.shl_const_u32x8::<24>(simd.shl_dyn_u32x8(
                                simd.and_u32x8(simd.splat_u32x8(*lhs as u32), MASK),
                                SHIFT,
                            ));
                            let acc = simd.xor_f32x8(cast(lhs), rhs);
                            dst_ = simd.add_f32x8(dst_, acc);
                            *dst = dst_;
                        }
                    }

                    if m % 8 != 0 {
                        let lhs = &lhs[m / 8];

                        let mut dst_ = simd.f32s_partial_load(dst_tail);
                        let lhs = simd.shl_const_u32x8::<24>(simd.shl_dyn_u32x8(
                            simd.and_u32x8(simd.splat_u32x8(*lhs as u32), MASK),
                            SHIFT,
                        ));
                        let acc = simd.xor_f32x8(cast(lhs), rhs);
                        dst_ = simd.add_f32x8(dst_, acc);
                        simd.f32s_partial_store(dst_tail, dst_);
                    }
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

pub fn matvec_f32_scalar(dst: &mut [f32], lhs: SignMatRef<'_>, rhs: &[f32]) {
    let m = lhs.nrows();
    let n = lhs.ncols();

    for j in 0..n {
        let rhs = rhs[j];
        let lhs = bytemuck::cast_slice::<_, u8>(lhs.storage().col_as_slice(j));

        for i in 0..m {
            let div = i / 8;
            let rem = i % 8;

            let lhs = (lhs[div] >> rem) & 1 == 1;
            dst[i] += if lhs { -rhs } else { rhs };
        }
    }
}

pub fn matvec_f32(dst: &mut [f32], lhs: SignMatRef<'_>, rhs: &[f32]) {
    assert!(all(lhs.nrows() == dst.len(), lhs.ncols() == rhs.len()));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(simd) = pulp::x86::V4::try_new() {
            return x86::matvec_f32_v4(simd, dst, lhs, rhs);
        }
        if let Some(simd) = pulp::x86::V3::try_new() {
            return x86::matvec_f32_v3(simd, dst, lhs, rhs);
        }
    }

    matvec_f32_scalar(dst, lhs, rhs)
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

            let rhs = (0..n).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();
            let acc = (0..m).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();

            let mut dst_actual = acc.clone();
            let mut dst_target = acc.clone();

            matvec_f32_scalar(&mut dst_target, lhs, &rhs);
            x86::matvec_f32_v4(simd, &mut dst_actual, lhs, &rhs);

            assert!(dst_actual == dst_target);
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

            let rhs = (0..n).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();
            let acc = (0..m).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();

            let mut dst_actual = acc.clone();
            let mut dst_target = acc.clone();

            matvec_f32_scalar(&mut dst_target, lhs, &rhs);
            x86::matvec_f32_v3(simd, &mut dst_actual, lhs, &rhs);

            assert!(dst_actual == dst_target);
        }
    }
}

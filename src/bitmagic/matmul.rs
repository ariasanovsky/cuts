use crate::{MatMut, SignMatRef};
use dyn_stack::PodStack;
use equator::assert;
use reborrow::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86 {
    use super::*;
    use dyn_stack::{GlobalPodBuffer, StackReq};
    use itertools::izip;
    use pulp::{cast, x86::V3, Simd};

    #[inline(always)]
    pub fn kernel_8x8_f32_v3(
        simd: V3,
        dst: MatMut<'_, f32>,
        lhs4: &[u32],
        rhs4: &[u32],
        diag: &[f32],
    ) {
        let k = diag.len();
        let div = k / 4;
        let rem = k % 4;

        use pulp::u32x8;
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

        let (diag4, diag) = pulp::as_arrays::<4, _>(diag);
        let mut acc = [[simd.splat_f32x8(0.0); 8]; 1];

        for (&lhs4, &rhs4, &diag4) in izip!(lhs4, rhs4, diag4) {
            let lhs4 = simd.shl_dyn_u32x8(simd.and_u32x8(simd.splat_u32x8(lhs4), MASK), SHIFT);
            let rhs4 = simd.splat_u32x8(rhs4);

            let d = [
                simd.splat_f32x8(diag4[0]),
                simd.splat_f32x8(diag4[1]),
                simd.splat_f32x8(diag4[2]),
                simd.splat_f32x8(diag4[3]),
            ];

            macro_rules! do_it {
                ($k: expr, $i: expr, $j: expr $(,)?) => {{
                    let lhs = simd.shl_const_u32x8::<{ 24 - $k * 8 }>(lhs4);
                    let rhs = simd.shl_const_u32x8::<{ 31 - ($k * 8 + $j) }>(rhs4);
                    let mul = simd.and_u32x8(simd.xor_u32x8(lhs, rhs), sign_bit);
                    acc[$i][$j] = simd.add_f32x8(acc[$i][$j], simd.xor_f32x8(cast(mul), d[$k]));
                }};
            }

            {
                do_it!(0, 0, 0);
                do_it!(0, 0, 1);
                do_it!(0, 0, 2);
                do_it!(0, 0, 3);
                do_it!(0, 0, 4);
                do_it!(0, 0, 5);
                do_it!(0, 0, 6);
                do_it!(0, 0, 7);
            }
            {
                do_it!(1, 0, 0);
                do_it!(1, 0, 1);
                do_it!(1, 0, 2);
                do_it!(1, 0, 3);
                do_it!(1, 0, 4);
                do_it!(1, 0, 5);
                do_it!(1, 0, 6);
                do_it!(1, 0, 7);
            }
            {
                do_it!(2, 0, 0);
                do_it!(2, 0, 1);
                do_it!(2, 0, 2);
                do_it!(2, 0, 3);
                do_it!(2, 0, 4);
                do_it!(2, 0, 5);
                do_it!(2, 0, 6);
                do_it!(2, 0, 7);
            }
            {
                do_it!(3, 0, 0);
                do_it!(3, 0, 1);
                do_it!(3, 0, 2);
                do_it!(3, 0, 3);
                do_it!(3, 0, 4);
                do_it!(3, 0, 5);
                do_it!(3, 0, 6);
                do_it!(3, 0, 7);
            }
        }

        if rem != 0 {
            let lhs4 = lhs4[div];
            let rhs4 = rhs4[div];

            let diag4 = [
                diag[0],
                diag.get(1).copied().unwrap_or(0.0),
                diag.get(2).copied().unwrap_or(0.0),
                diag.get(3).copied().unwrap_or(0.0),
            ];

            let lhs4 = simd.shl_dyn_u32x8(simd.and_u32x8(simd.splat_u32x8(lhs4), MASK), SHIFT);
            let rhs4 = simd.splat_u32x8(rhs4);
            let d = [
                simd.splat_f32x8(diag4[0]),
                simd.splat_f32x8(diag4[1]),
                simd.splat_f32x8(diag4[2]),
                simd.splat_f32x8(diag4[3]),
            ];

            macro_rules! do_it {
                ($k: expr, $i: expr, $j: expr $(,)?) => {{
                    let lhs = simd.shl_const_u32x8::<{ 24 - $k * 8 }>(lhs4);
                    let rhs = simd.shl_const_u32x8::<{ 31 - ($k * 8 + $j) }>(rhs4);
                    let mul = simd.and_u32x8(simd.xor_u32x8(lhs, rhs), sign_bit);
                    acc[$i][$j] = simd.add_f32x8(acc[$i][$j], simd.xor_f32x8(cast(mul), d[$k]));
                }};
            }

            {
                do_it!(0, 0, 0);
                do_it!(0, 0, 1);
                do_it!(0, 0, 2);
                do_it!(0, 0, 3);
                do_it!(0, 0, 4);
                do_it!(0, 0, 5);
                do_it!(0, 0, 6);
                do_it!(0, 0, 7);
            }
            {
                do_it!(1, 0, 0);
                do_it!(1, 0, 1);
                do_it!(1, 0, 2);
                do_it!(1, 0, 3);
                do_it!(1, 0, 4);
                do_it!(1, 0, 5);
                do_it!(1, 0, 6);
                do_it!(1, 0, 7);
            }
            {
                do_it!(2, 0, 0);
                do_it!(2, 0, 1);
                do_it!(2, 0, 2);
                do_it!(2, 0, 3);
                do_it!(2, 0, 4);
                do_it!(2, 0, 5);
                do_it!(2, 0, 6);
                do_it!(2, 0, 7);
            }
            {
                do_it!(3, 0, 0);
                do_it!(3, 0, 1);
                do_it!(3, 0, 2);
                do_it!(3, 0, 3);
                do_it!(3, 0, 4);
                do_it!(3, 0, 5);
                do_it!(3, 0, 6);
                do_it!(3, 0, 7);
            }
        }
        let mut dst = dst;
        macro_rules! do_it {
            ($i: expr, $j: expr) => {{
                let col = dst.rb_mut().col_as_slice_mut($j);
                let dst = simd.f32s_partial_load(col);
                simd.f32s_partial_store(col, simd.add_f32x8(dst, acc[$i][$j]));
            }};
        }

        if dst.ncols() == 8 {
            do_it!(0, 0);
            do_it!(0, 1);
            do_it!(0, 2);
            do_it!(0, 3);
            do_it!(0, 4);
            do_it!(0, 5);
            do_it!(0, 6);
            do_it!(0, 7);
        } else {
            for j in 0..dst.ncols() {
                do_it!(0, j);
            }
        }
    }

    pub fn mat_tmat_f32_v3(
        simd: V3,
        dst: MatMut<'_, f32>,
        lhs: SignMatRef<'_>,
        rhs: SignMatRef<'_>,
        diag: &[f32],
    ) {
        struct Impl<'a> {
            simd: V3,
            dst: MatMut<'a, f32>,
            lhs: SignMatRef<'a>,
            rhs: SignMatRef<'a>,
            diag: &'a [f32],
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
                    diag,
                } = self;

                let m = dst.nrows();
                let n = dst.ncols();
                let k = diag.len();

                let mc = 128usize;
                let nc = 1024usize;
                let kc = 256usize;

                let lhs = lhs.storage_as::<u8>();
                let rhs = rhs.storage_as::<u8>();

                let mut mem = GlobalPodBuffer::new(
                    StackReq::new_aligned::<u32>(
                        Ord::min(k, kc).div_ceil(4) * Ord::min(n, nc).next_multiple_of(8),
                        64,
                    )
                    .and(StackReq::new_aligned::<u32>(
                        Ord::min(k, kc).div_ceil(4) * Ord::min(m, mc).next_multiple_of(8),
                        64,
                    )),
                );
                let mut stack = PodStack::new(&mut mem);

                let mut acc = dst;

                let mut depth = 0;
                while depth < k {
                    let kb = Ord::min(kc, k - depth);
                    let diag = &diag[depth..][..kb];

                    let mut col = 0;
                    while col < n {
                        let nb = Ord::min(nc, n - col);

                        let rhs = rhs.submatrix(col / 8, depth, nb, kb);
                        let stride = kb.div_ceil(4);
                        let (packed_rhs, mut stack) = stack
                            .rb_mut()
                            .make_aligned_raw::<u32>(stride * nb.next_multiple_of(8), 64);

                        {
                            let mut col = 0;
                            while col * 8 < nb {
                                let packed_rhs = &mut packed_rhs[col * stride..][..stride];
                                let packed_rhs = bytemuck::cast_slice_mut::<u32, u8>(packed_rhs);

                                for depth in 0..kb {
                                    packed_rhs[depth] = rhs[(col, depth)];
                                }

                                col += 1;
                            }
                        }

                        let mut row = 0;
                        while row < m {
                            let mb = Ord::min(mc, m - row);

                            let lhs = lhs.submatrix(row / 8, depth, mb, kb);

                            let mut acc = acc.rb_mut().submatrix_mut(row, col, mb, nb);

                            let (packed_lhs, _) = stack
                                .rb_mut()
                                .make_aligned_raw::<u32>(stride * mb.next_multiple_of(8), 64);

                            {
                                let mut row = 0;
                                while row * 8 < mb {
                                    let packed_lhs = &mut packed_lhs[row * stride..][..stride];
                                    let packed_lhs =
                                        bytemuck::cast_slice_mut::<u32, u8>(packed_lhs);

                                    for depth in 0..kb {
                                        packed_lhs[depth] = lhs[(row, depth)];
                                    }

                                    row += 1;
                                }
                            }

                            let mut col = 0;
                            while col * 8 < nb {
                                let packed_rhs = &packed_rhs[col * stride..][..stride];

                                let mut row = 0;
                                while row * 8 < mb {
                                    let packed_lhs = &packed_lhs[row * stride..][..stride];
                                    let acc = acc.rb_mut().submatrix_mut(
                                        row * 8,
                                        col * 8,
                                        Ord::min(8, mb - row * 8),
                                        Ord::min(8, nb - col * 8),
                                    );

                                    kernel_8x8_f32_v3(simd, acc, packed_lhs, packed_rhs, diag);

                                    row += 1;
                                }
                                col += 1;
                            }

                            row += mb;
                        }
                        col += nb;
                    }
                    depth += kb;
                }
            }
        }

        simd.vectorize(Impl {
            simd,
            dst,
            lhs,
            rhs,
            diag,
        })
    }
}

pub fn mat_tmat_f32_scalar(
    acc: MatMut<'_, f32>,
    lhs: SignMatRef<'_>,
    rhs: SignMatRef<'_>,
    diag: &[f32],
) {
    let lhs = lhs.storage_as::<u8>();
    let rhs = rhs.storage_as::<u8>();

    let mut dst = acc;

    for i in 0..dst.nrows() {
        for j in 0..dst.ncols() {
            let mut acc = 0.0;

            for (k, &diag) in diag.iter().enumerate() {
                let lhs = (lhs[(i / 8, k)] >> (i % 8)) & 1 == 1;
                let rhs = (rhs[(j / 8, k)] >> (j % 8)) & 1 == 1;

                let mul = lhs ^ rhs;
                acc += if mul { -diag } else { diag };
            }

            dst[(i, j)] += acc;
        }
    }
}

#[track_caller]
pub fn mat_tmat_f32(dst: MatMut<'_, f32>, lhs: SignMatRef<'_>, rhs: SignMatRef<'_>, diag: &[f32]) {
    assert!(all(
        dst.nrows() == lhs.nrows(),
        dst.ncols() == rhs.nrows(),
        lhs.ncols() == rhs.ncols(),
        lhs.ncols() == diag.len(),
    ));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(simd) = pulp::x86::V3::try_new() {
            return x86::mat_tmat_f32_v3(simd, dst, lhs, rhs, diag);
        }
    }

    mat_tmat_f32_scalar(dst, lhs, rhs, diag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MatRef;
    use equator::assert;
    use rand::prelude::*;

    #[test]
    fn test_v3() {
        if let Some(simd) = pulp::x86::V3::try_new() {
            let m = 400usize;
            let n = 1300usize;
            let k = 7;

            let rng = &mut StdRng::seed_from_u64(0);

            let stride = m.div_ceil(64);
            let data = (0..k * stride)
                .map(|_| rng.gen::<u64>())
                .collect::<Vec<_>>();
            let lhs =
                SignMatRef::from_storage(MatRef::from_col_major_slice(&data, stride, k, stride), m);

            let stride = n.div_ceil(64);
            let data = (0..k * stride)
                .map(|_| rng.gen::<u64>())
                .collect::<Vec<_>>();
            let rhs =
                SignMatRef::from_storage(MatRef::from_col_major_slice(&data, stride, k, stride), n);

            let diag = &*(0..k).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();

            // let acc = (0..m * n).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();
            let acc = (0..m * n).map(|_| 0.0).collect::<Vec<_>>();

            let dst_actual = &mut *acc.clone();
            let dst_target = &mut *acc.clone();
            let mut dst_actual = MatMut::from_col_major_slice(dst_actual, m, n, m);
            let mut dst_target = MatMut::from_col_major_slice(dst_target, m, n, m);

            x86::mat_tmat_f32_v3(simd, dst_actual.rb_mut(), lhs, rhs, diag);
            mat_tmat_f32_scalar(dst_target.rb_mut(), lhs, rhs, diag);

            for i in 0..m {
                for j in 0..n {
                    let actual = dst_actual[(i, j)];
                    let target = dst_target[(i, j)];

                    assert!((actual - target).abs() <= 1e-4);
                }
            }
        }
    }
}

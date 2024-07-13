use aligned_vec::{avec, AVec};
use core::iter::zip;
use cuts_v2::SignMatRef;
use diol::prelude::*;
use equator::assert;
use half::bf16;
use pulp::{x86::V4, *};
use rand::prelude::*;

// S C T^top x

// T^top x
// x = [x0, y0, x1, y1, ...]
// x' <- [x0 + y0, x0 - y0, ...]
// x' = F x
// T^top x = (T^top F^inv) F x
// T^top x = 1/2 (T^top F) F x

fn tmatvec_i16(y: &mut [f32], max: f32, x: &[i16], t: &[u64], n: usize, chunk_size: usize) {
    struct Impl<'a> {
        simd: V4,
        y: &'a mut [f32],
        max: f32,
        x: &'a [i16],
        t: &'a [u64],
        n: usize,
        chunk_size: usize,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                y,
                max,
                x,
                t,
                n,
                chunk_size,
            } = self;

            assert!(n % 64 == 0);
            let chunk_size = usize::min(chunk_size * 2, n / 32);

            let x = pulp::as_arrays::<32, _>(x).0;
            let x: &[i16x32] = bytemuck::cast_slice(x);
            let zero = simd.splat_i16x32(0);
            let scale = max / 128.0;
            for (y, t) in y.chunks_exact_mut(4).zip(t.chunks_exact(4 * (n / 64))) {
                let t: &[[u32; 4]] = bytemuck::cast_slice(t);

                let mut acc0_f32 = 0.0f32;
                let mut acc1_f32 = 0.0f32;
                let mut acc2_f32 = 0.0f32;
                let mut acc3_f32 = 0.0f32;

                for (x, t) in x.chunks_exact(chunk_size).zip(t.chunks_exact(chunk_size)) {
                    let mut acc0a = zero;
                    let mut acc0b = zero;

                    let mut acc1a = zero;
                    let mut acc1b = zero;

                    let mut acc2a = zero;
                    let mut acc2b = zero;

                    let mut acc3a = zero;
                    let mut acc3b = zero;

                    for (&[xa, xb], &[ta, tb]) in
                        core::iter::zip(pulp::as_arrays(x).0, pulp::as_arrays(t).0)
                    {
                        acc0a = simd.select_i16x32(
                            b32(ta[0]),
                            simd.wrapping_sub_i16x32(acc0a, xa),
                            acc0a,
                        );
                        acc0b = simd.select_i16x32(
                            b32(tb[0]),
                            simd.wrapping_sub_i16x32(acc0b, xb),
                            acc0b,
                        );

                        acc1a = simd.select_i16x32(
                            b32(ta[1]),
                            simd.wrapping_sub_i16x32(acc1a, xa),
                            acc1a,
                        );
                        acc1b = simd.select_i16x32(
                            b32(tb[1]),
                            simd.wrapping_sub_i16x32(acc1b, xb),
                            acc1b,
                        );

                        acc2a = simd.select_i16x32(
                            b32(ta[2]),
                            simd.wrapping_sub_i16x32(acc2a, xa),
                            acc2a,
                        );
                        acc2b = simd.select_i16x32(
                            b32(tb[2]),
                            simd.wrapping_sub_i16x32(acc2b, xb),
                            acc2b,
                        );

                        acc3a = simd.select_i16x32(
                            b32(ta[3]),
                            simd.wrapping_sub_i16x32(acc3a, xa),
                            acc3a,
                        );
                        acc3b = simd.select_i16x32(
                            b32(tb[3]),
                            simd.wrapping_sub_i16x32(acc3b, xb),
                            acc3b,
                        );
                    }

                    let [acc0a0, acc0a1]: [i16x16; 2] = pulp::cast(acc0a);
                    let [acc0b0, acc0b1]: [i16x16; 2] = pulp::cast(acc0b);
                    let [acc1a0, acc1a1]: [i16x16; 2] = pulp::cast(acc1a);
                    let [acc1b0, acc1b1]: [i16x16; 2] = pulp::cast(acc1b);
                    let [acc2a0, acc2a1]: [i16x16; 2] = pulp::cast(acc2a);
                    let [acc2b0, acc2b1]: [i16x16; 2] = pulp::cast(acc2b);
                    let [acc3a0, acc3a1]: [i16x16; 2] = pulp::cast(acc3a);
                    let [acc3b0, acc3b1]: [i16x16; 2] = pulp::cast(acc3b);

                    let acc0a0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc0a0));
                    let acc0a1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc0a1));
                    let acc0a = simd.add_f32x16(acc0a0, acc0a1);
                    let acc0b0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc0b0));
                    let acc0b1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc0b1));
                    let acc0b = simd.add_f32x16(acc0b0, acc0b1);
                    let acc0 = simd.add_f32x16(acc0a, acc0b);
                    acc0_f32 += simd.f32s_reduce_sum(acc0);

                    let acc1a0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc1a0));
                    let acc1a1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc1a1));
                    let acc1a = simd.add_f32x16(acc1a0, acc1a1);
                    let acc1b0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc1b0));
                    let acc1b1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc1b1));
                    let acc1b = simd.add_f32x16(acc1b0, acc1b1);
                    let acc1 = simd.add_f32x16(acc1a, acc1b);
                    acc1_f32 += simd.f32s_reduce_sum(acc1);

                    let acc2a0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc2a0));
                    let acc2a1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc2a1));
                    let acc2a = simd.add_f32x16(acc2a0, acc2a1);
                    let acc2b0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc2b0));
                    let acc2b1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc2b1));
                    let acc2b = simd.add_f32x16(acc2b0, acc2b1);
                    let acc2 = simd.add_f32x16(acc2a, acc2b);
                    acc2_f32 += simd.f32s_reduce_sum(acc2);

                    let acc3a0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc3a0));
                    let acc3a1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc3a1));
                    let acc3a = simd.add_f32x16(acc3a0, acc3a1);
                    let acc3b0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc3b0));
                    let acc3b1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc3b1));
                    let acc3b = simd.add_f32x16(acc3b0, acc3b1);
                    let acc3 = simd.add_f32x16(acc3a, acc3b);
                    acc3_f32 += simd.f32s_reduce_sum(acc3);
                }

                y[0] = acc0_f32 * scale;
                y[1] = acc1_f32 * scale;
                y[2] = acc2_f32 * scale;
                y[3] = acc3_f32 * scale;
            }
        }
    }

    let simd = V4::try_new().unwrap();
    simd.vectorize(Impl {
        simd,
        y,
        max,
        x,
        t,
        n,
        chunk_size,
    })
}

fn tmatvec_i8(y: &mut [f32], max: f32, x: &[i8], t: &[u64], n: usize, chunk_size: usize) {
    struct Impl<'a> {
        simd: V4,
        y: &'a mut [f32],
        max: f32,
        x: &'a [i8],
        t: &'a [u64],
        n: usize,
        chunk_size: usize,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                y,
                max,
                x,
                t,
                n,
                chunk_size,
            } = self;

            assert!(n % 64 == 0);
            let chunk_size = usize::min(chunk_size * 2, n / 32);

            let x = pulp::as_arrays::<64, _>(x).0;
            let x: &[i8x64] = bytemuck::cast_slice(x);
            let zero = simd.splat_i8x64(0);
            let scale = max / 128.0;
            for (y, t) in y.chunks_exact_mut(4).zip(t.chunks_exact(4 * (n / 64))) {
                let t: &[[u64; 4]] = bytemuck::cast_slice(t);

                let mut acc0_f32 = 0.0f32;
                let mut acc1_f32 = 0.0f32;
                let mut acc2_f32 = 0.0f32;
                let mut acc3_f32 = 0.0f32;

                for (x, t) in x.chunks_exact(chunk_size).zip(t.chunks_exact(chunk_size)) {
                    let mut acc0a = zero;
                    let mut acc0b = zero;

                    let mut acc1a = zero;
                    let mut acc1b = zero;

                    let mut acc2a = zero;
                    let mut acc2b = zero;

                    let mut acc3a = zero;
                    let mut acc3b = zero;

                    for (&[xa, xb], &[ta, tb]) in
                        core::iter::zip(pulp::as_arrays(x).0, pulp::as_arrays(t).0)
                    {
                        acc0a = simd.select_i8x64(
                            b64(ta[0]),
                            simd.wrapping_sub_i8x64(acc0a, xa),
                            acc0a,
                        );
                        acc0b = simd.select_i8x64(
                            b64(tb[0]),
                            simd.wrapping_sub_i8x64(acc0b, xb),
                            acc0b,
                        );

                        acc1a = simd.select_i8x64(
                            b64(ta[1]),
                            simd.wrapping_sub_i8x64(acc1a, xa),
                            acc1a,
                        );
                        acc1b = simd.select_i8x64(
                            b64(tb[1]),
                            simd.wrapping_sub_i8x64(acc1b, xb),
                            acc1b,
                        );

                        acc2a = simd.select_i8x64(
                            b64(ta[2]),
                            simd.wrapping_sub_i8x64(acc2a, xa),
                            acc2a,
                        );
                        acc2b = simd.select_i8x64(
                            b64(tb[2]),
                            simd.wrapping_sub_i8x64(acc2b, xb),
                            acc2b,
                        );

                        acc3a = simd.select_i8x64(
                            b64(ta[3]),
                            simd.wrapping_sub_i8x64(acc3a, xa),
                            acc3a,
                        );
                        acc3b = simd.select_i8x64(
                            b64(tb[3]),
                            simd.wrapping_sub_i8x64(acc3b, xb),
                            acc3b,
                        );
                    }

                    let [acc0a0, acc0a1]: [i16x16; 2] = pulp::cast(acc0a);
                    let [acc0b0, acc0b1]: [i16x16; 2] = pulp::cast(acc0b);
                    let [acc1a0, acc1a1]: [i16x16; 2] = pulp::cast(acc1a);
                    let [acc1b0, acc1b1]: [i16x16; 2] = pulp::cast(acc1b);
                    let [acc2a0, acc2a1]: [i16x16; 2] = pulp::cast(acc2a);
                    let [acc2b0, acc2b1]: [i16x16; 2] = pulp::cast(acc2b);
                    let [acc3a0, acc3a1]: [i16x16; 2] = pulp::cast(acc3a);
                    let [acc3b0, acc3b1]: [i16x16; 2] = pulp::cast(acc3b);

                    let acc0a0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc0a0));
                    let acc0a1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc0a1));
                    let acc0a = simd.add_f32x16(acc0a0, acc0a1);
                    let acc0b0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc0b0));
                    let acc0b1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc0b1));
                    let acc0b = simd.add_f32x16(acc0b0, acc0b1);
                    let acc0 = simd.add_f32x16(acc0a, acc0b);
                    acc0_f32 += simd.f32s_reduce_sum(acc0);

                    let acc1a0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc1a0));
                    let acc1a1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc1a1));
                    let acc1a = simd.add_f32x16(acc1a0, acc1a1);
                    let acc1b0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc1b0));
                    let acc1b1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc1b1));
                    let acc1b = simd.add_f32x16(acc1b0, acc1b1);
                    let acc1 = simd.add_f32x16(acc1a, acc1b);
                    acc1_f32 += simd.f32s_reduce_sum(acc1);

                    let acc2a0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc2a0));
                    let acc2a1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc2a1));
                    let acc2a = simd.add_f32x16(acc2a0, acc2a1);
                    let acc2b0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc2b0));
                    let acc2b1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc2b1));
                    let acc2b = simd.add_f32x16(acc2b0, acc2b1);
                    let acc2 = simd.add_f32x16(acc2a, acc2b);
                    acc2_f32 += simd.f32s_reduce_sum(acc2);

                    let acc3a0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc3a0));
                    let acc3a1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc3a1));
                    let acc3a = simd.add_f32x16(acc3a0, acc3a1);
                    let acc3b0 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc3b0));
                    let acc3b1 =
                        simd.convert_i32x16_to_f32x16(simd.convert_i16x16_to_i32x16(acc3b1));
                    let acc3b = simd.add_f32x16(acc3b0, acc3b1);
                    let acc3 = simd.add_f32x16(acc3a, acc3b);
                    acc3_f32 += simd.f32s_reduce_sum(acc3);
                }

                y[0] = acc0_f32 * scale;
                y[1] = acc1_f32 * scale;
                y[2] = acc2_f32 * scale;
                y[3] = acc3_f32 * scale;
            }
        }
    }

    let simd = V4::try_new().unwrap();
    simd.vectorize(Impl {
        simd,
        y,
        max,
        x,
        t,
        n,
        chunk_size,
    })
}

fn matvec_i16(
    y_i16: &mut [i16],
    y: &mut [f32],
    max: f32,
    x: &[i16],
    s: &[u64],
    m: usize,
    chunk_size: usize,
) {
    struct Impl<'a> {
        simd: V4,
        y_i16: &'a mut [i16],
        y: &'a mut [f32],
        max: f32,
        x: &'a [i16],
        s: &'a [u64],
        m: usize,
        chunk_size: usize,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                y_i16,
                y,
                max,
                x,
                s,
                m,
                chunk_size,
            } = self;

            assert!(m % 64 == 0);

            let zero = simd.splat_i16x32(0);
            for (x, s) in x
                .chunks_exact(chunk_size)
                .zip(s.chunks_exact(chunk_size * (m / 64)))
            {
                {
                    let y_i16 = pulp::as_arrays_mut::<32, _>(y_i16).0;
                    let y_i16: &mut [i16x32] = bytemuck::cast_slice_mut(y_i16);
                    y_i16.fill(zero);

                    for (x, s) in x.chunks_exact(4).zip(s.chunks_exact(4 * (m / 64))) {
                        let s: &[[b32; 4]] = bytemuck::cast_slice(s);
                        let x0 = simd.splat_i16x32(x[0]);
                        let x1 = simd.splat_i16x32(x[1]);
                        let x2 = simd.splat_i16x32(x[2]);
                        let x3 = simd.splat_i16x32(x[3]);

                        for (y, &[s0, s1, s2, s3]) in y_i16.iter_mut().zip(s) {
                            let mut t = *y;

                            t = simd.select_i16x32(s0, simd.wrapping_sub_i16x32(t, x0), t);
                            t = simd.select_i16x32(s1, simd.wrapping_sub_i16x32(t, x1), t);
                            t = simd.select_i16x32(s2, simd.wrapping_sub_i16x32(t, x2), t);
                            t = simd.select_i16x32(s3, simd.wrapping_sub_i16x32(t, x3), t);

                            *y = t;
                        }
                    }
                }

                for (y, &y_i16) in y.iter_mut().zip(&*y_i16) {
                    *y += y_i16 as f32;
                }
            }

            let max = max / 128.0;
            for y in y.iter_mut() {
                *y = *y * max;
            }
        }
    }

    let simd = V4::try_new().unwrap();
    simd.vectorize(Impl {
        simd,
        y_i16,
        y,
        max,
        x,
        s,
        m,
        chunk_size,
    })
}

fn matvec_i8(
    y_i8: &mut [i8],
    y: &mut [f32],
    max: f32,
    x: &[i8],
    s: &[u64],
    m: usize,
    chunk_size: usize,
) {
    struct Impl<'a> {
        simd: V4,
        y_i8: &'a mut [i8],
        y: &'a mut [f32],
        max: f32,
        x: &'a [i8],
        s: &'a [u64],
        m: usize,
        chunk_size: usize,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                y_i8,
                y,
                max,
                x,
                s,
                m,
                chunk_size,
            } = self;

            assert!(m % 64 == 0);

            let zero = simd.splat_i8x64(0);
            for (x, s) in x
                .chunks_exact(chunk_size)
                .zip(s.chunks_exact(chunk_size * (m / 64)))
            {
                {
                    let y_i8 = pulp::as_arrays_mut::<64, _>(y_i8).0;
                    let y_i8: &mut [i8x64] = bytemuck::cast_slice_mut(y_i8);
                    y_i8.fill(zero);

                    for (x, s) in x.chunks_exact(4).zip(s.chunks_exact(4 * (m / 64))) {
                        let s: &[[b64; 4]] = bytemuck::cast_slice(s);
                        let x0 = simd.splat_i8x64(x[0]);
                        let x1 = simd.splat_i8x64(x[1]);
                        let x2 = simd.splat_i8x64(x[2]);
                        let x3 = simd.splat_i8x64(x[3]);

                        for (y, &[s0, s1, s2, s3]) in y_i8.iter_mut().zip(s) {
                            let mut t = *y;

                            t = simd.select_i8x64(s0, simd.wrapping_sub_i8x64(t, x0), t);
                            t = simd.select_i8x64(s1, simd.wrapping_sub_i8x64(t, x1), t);
                            t = simd.select_i8x64(s2, simd.wrapping_sub_i8x64(t, x2), t);
                            t = simd.select_i8x64(s3, simd.wrapping_sub_i8x64(t, x3), t);

                            *y = t;
                        }
                    }
                }

                for (y, &y_i8) in y.iter_mut().zip(&*y_i8) {
                    *y += y_i8 as f32;
                }
            }

            let max = max / 128.0;
            for y in y.iter_mut() {
                *y = *y * max;
            }
        }
    }

    let simd = V4::try_new().unwrap();
    simd.vectorize(Impl {
        simd,
        y_i8,
        y,
        max,
        x,
        s,
        m,
        chunk_size,
    })
}

fn sctvec_i16(bencher: Bencher, (m, n, compression): (usize, usize, f64)) {
    let width = (compression * (m * n * 2) as f64
        / (m.next_multiple_of(64) / 8 + n.next_multiple_of(64) / 8 + 4) as f64)
        as usize;

    let ref mut rng = StdRng::seed_from_u64(0);

    let chunk_size = 256;
    let width = width.next_multiple_of(chunk_size);

    let x = &*AVec::<_>::from_iter(0, (0..n).map(|_| (rng.gen::<f32>() - 0.5)));
    let x_i16 = &mut *avec![0i16; n];
    let y = &mut *avec![0.0f32; width];
    let y_i16 = &mut *avec![0i16; width];
    let z = &mut *avec![0.0f32; m];
    let z_i16 = &mut *avec![0i16; m];

    let c = &*avec![1.0f32; width];
    let t = &*avec![u64::MAX;  n.div_ceil(64) * width];
    let s = &*avec![u64::MAX;  m.div_ceil(64) * width];

    bencher.bench(|| {
        {
            let max_exponent = x
                .iter()
                .map(|&x| x.abs().log2().ceil() as i32)
                .max()
                .unwrap();
            let max = 2.0f32.powi(max_exponent);

            for (x_i16, x) in x_i16.chunks_mut(chunk_size).zip(x.chunks(chunk_size)) {
                for (x_i16, &x) in x_i16.iter_mut().zip(x) {
                    *x_i16 = ((x / max) * 128.0) as i16;
                }
            }

            tmatvec_i16(y, max, x_i16, t, n, chunk_size);
        }

        for (y, &c) in y.iter_mut().zip(c) {
            *y = *y * c;
        }

        {
            let max_exponent = y
                .iter()
                .map(|&y| y.abs().log2().ceil() as i32)
                .max()
                .unwrap();
            let max = 2.0f32.powi(max_exponent);

            for (y_i16, y) in y_i16.chunks_mut(chunk_size).zip(y.chunks(chunk_size)) {
                for (y_i16, &y) in y_i16.iter_mut().zip(y) {
                    *y_i16 = ((y / max) * 128.0) as i16;
                }
            }

            matvec_i16(z_i16, z, max, y_i16, s, m, chunk_size);
        }
    });
}

fn sctvec_i8(bencher: Bencher, (m, n, compression): (usize, usize, f64)) {
    let width = (compression * (m * n * 2) as f64
        / (m.next_multiple_of(64) / 8 + n.next_multiple_of(64) / 8 + 4) as f64)
        as usize;
    let width = width.next_multiple_of(64);

    let ref mut rng = StdRng::seed_from_u64(0);

    let chunk_size = 16;

    let x = &*AVec::<_>::from_iter(0, (0..n).map(|_| (rng.gen::<f32>() - 0.5)));
    let x_i8 = &mut *avec![0i8; n];
    let y = &mut *avec![0.0f32; width];
    let y_i8 = &mut *avec![0i8; width];
    let z = &mut *avec![0.0f32; m];
    let z_i8 = &mut *avec![0i8; m];

    let c = &*avec![1.0f32; width];
    let t = &*avec![u64::MAX;  n.div_ceil(64) * width];
    let s = &*avec![u64::MAX;  m.div_ceil(64) * width];

    bencher.bench(|| {
        {
            let max_exponent = x
                .iter()
                .map(|&x| x.abs().log2().ceil() as i32)
                .max()
                .unwrap();
            let max = 2.0f32.powi(max_exponent);

            for (x_i8, x) in x_i8.chunks_mut(chunk_size).zip(x.chunks(chunk_size)) {
                for (x_i8, &x) in x_i8.iter_mut().zip(x) {
                    *x_i8 = ((x / max) * 16.0) as i8;
                }
            }

            tmatvec_i8(y, max, x_i8, t, n, chunk_size);
        }

        for (y, &c) in y.iter_mut().zip(c) {
            *y = *y * c;
        }

        {
            let max_exponent = y
                .iter()
                .map(|&y| y.abs().log2().ceil() as i32)
                .max()
                .unwrap();
            let max = 2.0f32.powi(max_exponent);

            for (y_i8, y) in y_i8.chunks_mut(chunk_size).zip(y.chunks(chunk_size)) {
                for (y_i8, &y) in y_i8.iter_mut().zip(y) {
                    *y_i8 = ((y / max) * 16.0) as i8;
                }
            }

            matvec_i8(z_i8, z, max, y_i8, s, m, chunk_size);
        }
    });
}

fn sctvec_f32(
    bencher: Bencher,
    (m, n, compression, transpose_s, transpose_t): (usize, usize, f64, bool, bool),
) {
    // sct_size = (nrows.next_multiple_of(64) / 8 + ncols.next_multiple_of(64) / 8 + 4) * width
    // bf16_size = nrows * ncols * 2
    // sct_size = compression_rate * bf16_size
    let width = (compression * (m * n * 2) as f64
        / (m.next_multiple_of(64) / 8 + n.next_multiple_of(64) / 8 + 4) as f64)
        as usize;

    let c = &*avec![1.0f32; width];
    let s =
        &*avec![u64::MAX; if transpose_s {width.div_ceil(64) * m} else {m.div_ceil(64) * width}];
    let t =
        &*avec![u64::MAX; if transpose_t {width.div_ceil(64) * n} else {n.div_ceil(64) * width}];

    let s = if transpose_s {
        SignMatRef::from_storage(
            cuts_v2::MatRef::from_col_major_slice(s, width.div_ceil(64), m, width.div_ceil(64)),
            width,
        )
    } else {
        SignMatRef::from_storage(
            cuts_v2::MatRef::from_col_major_slice(s, m.div_ceil(64), width, m.div_ceil(64)),
            m,
        )
    };
    let t = if transpose_t {
        SignMatRef::from_storage(
            cuts_v2::MatRef::from_col_major_slice(t, width.div_ceil(64), n, width.div_ceil(64)),
            width,
        )
    } else {
        SignMatRef::from_storage(
            cuts_v2::MatRef::from_col_major_slice(t, n.div_ceil(64), width, n.div_ceil(64)),
            n,
        )
    };

    let x = &*avec![1.0f32; n];
    let tmp = &mut *avec![1.0f32; width];
    let y = &mut *avec![1.0f32; m];

    bencher.bench(|| {
        tmp.fill(0.0);
        y.fill(0.0);
        if transpose_t {
            cuts_v2::bitmagic::matvec::matvec_f32(tmp, t, x);
        } else {
            cuts_v2::bitmagic::tmatvec::tmatvec_f32(tmp, t, x);
        }
        for (x, &c) in zip(&mut *tmp, c) {
            *x = c * *x;
        }
        if transpose_s {
            cuts_v2::bitmagic::tmatvec::tmatvec_f32(y, s, tmp);
        } else {
            cuts_v2::bitmagic::matvec::matvec_f32(y, s, tmp);
        }
    })
}

// fn sctvec_f32_nn_1(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec_f32(bencher, (m, n, 1.0, false, false));
// }
// fn sctvec_f32_nn_0_5(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec_f32(bencher, (m, n, 0.5, false, false));
// }
// fn sctvec_f32_nn_0_25(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec_f32(bencher, (m, n, 0.25, false, false));
// }

fn sctvec_i16_nn_1(bencher: Bencher, (m, n): (usize, usize)) {
    sctvec_i16(bencher, (m, n, 1.0));
}
fn sctvec_i16_nn_0_5(bencher: Bencher, (m, n): (usize, usize)) {
    sctvec_i16(bencher, (m, n, 0.5));
}
fn sctvec_i16_nn_0_25(bencher: Bencher, (m, n): (usize, usize)) {
    sctvec_i16(bencher, (m, n, 0.25));
}

fn sctvec_i8_nn_1(bencher: Bencher, (m, n): (usize, usize)) {
    sctvec_i8(bencher, (m, n, 1.0));
}
fn sctvec_i8_nn_0_5(bencher: Bencher, (m, n): (usize, usize)) {
    sctvec_i8(bencher, (m, n, 0.5));
}
fn sctvec_i8_nn_0_25(bencher: Bencher, (m, n): (usize, usize)) {
    sctvec_i8(bencher, (m, n, 0.25));
}

// fn sctvec_tn_1(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec(bencher, (m, n, 1.0, true, false));
// }
// fn sctvec_tn_0_5(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec(bencher, (m, n, 0.5, true, false));
// }
// fn sctvec_tn_0_25(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec(bencher, (m, n, 0.25, true, false));
// }

// fn sctvec_nt_1(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec(bencher, (m, n, 1.0, false, true));
// }
// fn sctvec_nt_0_5(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec(bencher, (m, n, 0.5, false, true));
// }
// fn sctvec_nt_0_25(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec(bencher, (m, n, 0.25, false, true));
// }

// fn sctvec_tt_1(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec(bencher, (m, n, 1.0, true, true));
// }
// fn sctvec_tt_0_5(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec(bencher, (m, n, 0.5, true, true));
// }
// fn sctvec_tt_0_25(bencher: Bencher, (m, n): (usize, usize)) {
//     sctvec(bencher, (m, n, 0.25, true, true));
// }

fn matvec_f32(bencher: Bencher, (m, n): (usize, usize)) {
    let stride = m.next_multiple_of(16);
    let a = &*avec![1.0f32; stride * n];

    let x = &*avec![1.0f32; n];
    let y = &mut *avec![1.0f32; m];

    bencher.bench(|| {
        struct Impl<'a> {
            stride: usize,
            a: &'a [f32],
            x: &'a [f32],
            y: &'a mut [f32],
        }

        impl pulp::WithSimd for Impl<'_> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, _: S) -> Self::Output {
                let Self { stride, a, x, y } = self;
                y.fill(0.0);

                for (a, x) in zip(a.chunks_exact(stride), x) {
                    let x = *x;
                    for (y, &a) in zip(&mut *y, a) {
                        *y = f32::mul_add(a, x, *y);
                    }
                }
            }
        }

        pulp::Arch::new().dispatch(Impl { stride, a, x, y });
    })
}

fn matvec_bf16(bencher: Bencher, (m, n): (usize, usize)) {
    let stride = m.next_multiple_of(16);
    let a = &*avec![bf16::ONE; stride * n];
    let x = &*avec![1.0f32; n];
    let y = &mut *avec![1.0f32; m];

    bencher.bench(|| {
        struct Impl<'a> {
            stride: usize,
            a: &'a [bf16],
            x: &'a [f32],
            y: &'a mut [f32],
        }

        impl pulp::WithSimd for Impl<'_> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: Simd>(self, _: S) -> Self::Output {
                let Self { stride, a, x, y } = self;
                y.fill(0.0);

                for (a, x) in zip(a.chunks_exact(stride), x) {
                    let x = *x;
                    for (y, &a) in zip(&mut *y, a) {
                        let a = a.to_bits();
                        let a = (a as u32) << 16;
                        let a = f32::from_bits(a);
                        *y = f32::mul_add(a, x, *y);
                    }
                }
            }
        }

        pulp::Arch::new().dispatch(Impl { stride, a, x, y });
    })
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);

    bench.register_many(
        list![
            // sctvec_f32_nn_1,
            sctvec_i16_nn_1,
            sctvec_i8_nn_1,
            //
            // sctvec_f32_nn_0_5,
            sctvec_i16_nn_0_5,
            sctvec_i8_nn_0_5,
            //
            // sctvec_f32_nn_0_25,
            sctvec_i16_nn_0_25,
            sctvec_i8_nn_0_25,
            //
            matvec_f32,
            matvec_bf16,
        ],
        [
            //
            (1 * 1024, 1 * 1024),
            (2 * 1024, 2 * 1024),
            (3 * 1024, 3 * 1024),
            (4 * 1024, 4 * 1024),
            (8 * 1024, 8 * 1024),
        ],
    );

    bench.run()?;

    Ok(())
}

use aligned_vec::{avec, ABox};
use core::{cell::Cell, iter};
use dyn_stack::PodStack;
use equator::assert;
use rand::Rng;
use reborrow::*;

use crate::inplace_sct::sparse_matvec;

pub struct Remainder {
    dim: Box<[usize]>,
    total_dim: usize,
    t: ABox<[f32]>,
    two_mats: Box<[faer::Mat<f32>]>,
}

impl Remainder {
    pub fn new(tensor: &[f32], dim: &[usize], stride: &[usize]) -> Self {
        let order = dim.len();
        let total_dim = dim.iter().product::<usize>();
        assert!(total_dim != 0);

        let mut t = avec![0.0f32; total_dim].into_boxed_slice();
        let mut state = vec![0usize; order].into_boxed_slice();
        let mut offset = vec![0usize; order].into_boxed_slice();

        let mut dst_index = 0usize;
        'outer: loop {
            let mut src_index = 0usize;
            for &offset in &*offset {
                src_index += offset;
            }

            t[dst_index] = tensor[src_index];

            dst_index += 1;

            let mut pos = 0usize;
            'inner: loop {
                if pos == order {
                    break 'outer;
                }

                state[pos] += 1;
                offset[pos] += stride[pos];
                let carry = state[pos] == dim[pos];

                if carry {
                    state[pos] = 0;
                    offset[pos] = 0;
                    pos += 1;
                } else {
                    break 'inner;
                }
            }
        }

        Self {
            dim: dim.to_vec().into_boxed_slice(),
            total_dim,
            t,
            two_mats: dim
                .iter()
                .map(|&dim| faer::Mat::zeros(dim, total_dim / dim))
                .collect(),
        }
    }

    pub fn norm_l2(&self) -> f32 {
        faer::col::from_slice(&self.t).norm_l2()
    }

    pub fn fill_matrices(&mut self) {
        {
            let dim = &*self.dim;
            let order = dim.len();

            let mut dim_acc = vec![0usize; order].into_boxed_slice();

            let mut acc = 1;
            for (dim_acc, dim) in iter::zip(&mut dim_acc, dim) {
                *dim_acc = acc;
                acc *= dim;
            }

            let mut state = vec![0usize; order - 1].into_boxed_slice();
            let mut offset = vec![0usize; order - 1].into_boxed_slice();
            let mut stride = vec![0usize; order - 1].into_boxed_slice();
            let mut xdim = vec![0usize; order - 1].into_boxed_slice();

            for axis in 0..order {
                let t = &mut self.two_mats[axis];
                let tensor = &*self.t;

                stride.fill(0);
                xdim.fill(0);

                let mut s = 1usize;
                for i in 0..order {
                    if i < axis {
                        stride[i] = s;
                        xdim[i] = dim[i];
                    } else if i > axis {
                        stride[i - 1] = s;
                        xdim[i - 1] = dim[i];
                    }
                    s *= dim[i];
                }

                let mut dst_col = 0usize;

                state.fill(0);
                offset.fill(0);

                let axis_stride = dim_acc[axis];

                'outer: loop {
                    let mut src_index = 0;
                    for &offset in &*offset {
                        src_index += offset;
                    }

                    {
                        let t = t.as_mut().col_mut(dst_col);
                        let tensor = &tensor[src_index..];

                        for (dst, src) in
                            iter::zip(t.iter_mut(), tensor.iter().step_by(axis_stride))
                        {
                            *dst = 2.0 * src;
                        }
                    }
                    dst_col += 1;

                    let mut pos = 0usize;
                    'inner: loop {
                        if pos == order - 1 {
                            break 'outer;
                        }

                        state[pos] += 1;
                        offset[pos] += stride[pos];
                        let carry = state[pos] == xdim[pos];

                        if carry {
                            state[pos] = 0;
                            offset[pos] = 0;
                            pos += 1;
                        } else {
                            break 'inner;
                        }
                    }
                }
            }
        }
    }

    pub fn cut(&self, cut: &mut Cut, rng: &mut impl Rng, stack: PodStack<'_>) {
        let mut stack = stack;
        cut.setup(self, rng, stack.rb_mut());
        loop {
            let mut improved = false;
            for axis in 0..self.dim.len() {
                improved |= cut.improve(axis, self, stack.rb_mut());
            }
            if !improved {
                return;
            }
        }
    }

    pub fn update(&mut self, cut: &Cut, stack: PodStack<'_>) {
        let scale = -cut.c / self.total_dim as f32;
        cut.blowup_mul_add(self.t.as_mut(), scale, stack);
    }

    pub fn width_bits(&self) -> usize {
        32 + self.dim.iter().sum::<usize>()
    }
}

pub struct Cut {
    total_dim: usize,
    dim: Box<[usize]>,
    s_old: Box<[Box<[u64]>]>,
    s: Box<[Box<[u64]>]>,
    s_image: Box<[ABox<[f32]>]>,
    c: f32,
}

fn bit_kron(dst: &mut [u64], lhs: &[u64], lhs_nbits: usize, rhs: &[u64], rhs_nbits: usize) {
    let src_bits = rhs_nbits;
    let src_limbs = src_bits.div_ceil(64);
    let src = &rhs[..src_limbs];
    let (src_last_limb_bits, src_last_limb_mask) = if src_bits % 64 == 0 {
        (64, !0u64)
    } else {
        let bits = src_bits % 64;
        (bits, (1u64 << bits) - 1)
    };

    if src_bits == 0 {
        return;
    }

    for i in 0..lhs_nbits {
        let bit = (lhs[i / 64] >> (i % 64)) & 1 != 0;
        let splat_bit: u64 = if bit { !0 } else { 0 };

        let offset = i * src_bits;
        let limb_offset = offset / 64;
        let bit_offset = offset % 64;

        if bit_offset == 0 {
            let (&src_last, src) = src.split_last().unwrap();

            for (dst, &src) in iter::zip(&mut dst[limb_offset..][..src_limbs - 1], src) {
                *dst = src ^ splat_bit;
            }

            let src = (src_last ^ splat_bit) & src_last_limb_mask;
            dst[limb_offset + (src_limbs - 1)] = src;
        } else {
            // assume BITS = 4, bit_offset = 1

            // dst = [xxxy][yyyx]
            // src = vvvv

            // lo_mask = [1110]
            let lo_mask = (1u64 << bit_offset) - 1;
            let (&src_last, src) = src.split_last().unwrap();

            for (dst, &src) in iter::zip(
                Cell::from_mut(&mut dst[limb_offset..][..src_limbs])
                    .as_slice_of_cells()
                    .windows(2),
                src,
            ) {
                let [lo, hi] = dst else { panic!() };
                let src = src ^ splat_bit;

                // least significant `bit_offset` bits are kept
                // rest are taken from value

                // lo.get() = [xxxy]
                // lo.get() & lo_mask = [xxx0]
                // src << bit_offset  = [000v]
                lo.set((lo.get() & lo_mask) | (src << bit_offset));

                // src >> (BITS - bit_offset)  = [vvv0]
                hi.set(src >> (64 - bit_offset));
            }

            let src = (src_last ^ splat_bit) & src_last_limb_mask;
            let lo = &mut dst[limb_offset + (src_limbs - 1)];
            *lo = (*lo & lo_mask) | (src << bit_offset);

            let written_bits = (src_limbs - 1) * 64 + Ord::min(src_last_limb_bits, 64 - bit_offset);
            if written_bits < src_bits {
                let hi = &mut dst[limb_offset + src_limbs];
                *hi = src >> (64 - bit_offset);
            }
        }
    }
}

impl Cut {
    pub fn new<'a>(
        dim: &[usize],
        two_mats: impl IntoIterator<Item = faer::MatRef<'a, f32>>,
        stack: PodStack<'_>,
    ) -> Self {
        let mut stack = stack;
        let s = dim
            .iter()
            .map(|&dim| vec![0u64; dim.div_ceil(64)].into_boxed_slice())
            .collect::<Box<[_]>>();
        let total_dim = dim.iter().product();
        Self {
            dim: dim.to_vec().into_boxed_slice(),
            total_dim,
            s_old: dim
                .iter()
                .enumerate()
                .map(|(axis, _)| {
                    let n = total_dim / dim[axis];
                    let mut s_old = vec![0u64; n.div_ceil(64)].into_boxed_slice();
                    Self::blowup_along(&mut s_old, axis, dim, &s, stack.rb_mut());
                    s_old
                })
                .collect(),
            s,
            s_image: two_mats
                .into_iter()
                .map(|two_m| avec![0.0_f32; two_m.nrows()].into_boxed_slice())
                .collect(),
            c: f32::NEG_INFINITY,
        }
    }

    fn update_image(&mut self, axis: usize, remainder: &Remainder, stack: PodStack<'_>) {
        let mut stack = stack;
        let n = self.total_dim / self.dim[axis];
        let (s_new, mut stack) = stack.rb_mut().make_raw::<u64>(n.div_ceil(64));
        Self::blowup_along(s_new, axis, &self.dim, &self.s, stack.rb_mut());

        let s_old = &mut *self.s_old[axis];

        let (diff_indices, _) = stack.make_raw::<u64>(n);
        let mut pos = 0usize;
        for j in 0..n {
            let s_neg = (s_new[j / 64] >> (j % 64)) & 1 == 1;
            let s_neg_old = (s_old[j / 64] >> (j % 64)) & 1 == 1;

            if s_neg != s_neg_old {
                diff_indices[pos] = ((s_neg as u64) << 63) | j as u64;
                pos += 1;
            }
        }
        let diff_indices = &diff_indices[..pos];

        sparse_matvec(
            &mut self.s_image[axis],
            remainder.two_mats[axis].as_ref(),
            diff_indices,
        );
        s_old.copy_from_slice(s_new);
    }

    pub fn setup(&mut self, remainder: &Remainder, rng: &mut impl Rng, stack: PodStack<'_>) {
        self.c = f32::NEG_INFINITY;
        self.s.iter_mut().flatten().for_each(|s| *s = rng.gen());

        let mut stack = stack;
        for axis in 0..self.dim.len() {
            let n = self.total_dim / self.dim[axis];
            let s = &mut self.s_old[axis];
            Self::blowup_along(s, axis, &self.dim, &self.s, stack.rb_mut());

            {
                let (diff_indices, _) = stack.rb_mut().make_raw::<u64>(n);
                let mut pos = 0usize;
                for j in 0..n {
                    let s_neg = (s[j / 64] >> (j % 64)) & 1 == 1;
                    diff_indices[pos] = ((s_neg as u64) << 63) | j as u64;
                    pos += 1;
                }
                let diff_indices = &diff_indices[..pos];

                self.s_image[axis].fill(0.0);
                sparse_matvec(
                    &mut self.s_image[axis],
                    remainder.two_mats[axis].as_ref(),
                    diff_indices,
                );
            }

            for x in &mut *self.s_image[axis] {
                *x = *x * 0.5;
            }
        }
    }

    pub fn dim(&self, axis: usize) -> usize {
        self.dim[axis]
    }

    pub fn improve(&mut self, axis: usize, remainder: &Remainder, stack: PodStack<'_>) -> bool {
        let mut stack = stack;
        let n = self.total_dim / self.dim(axis);
        let (s, mut stack) = stack.rb_mut().make_raw::<u64>(n.div_ceil(64));
        Self::blowup_along(s, axis, &self.dim, &self.s, stack.rb_mut());
        self.update_image(axis, remainder, stack.rb_mut());

        let s_image = &*self.s_image[axis];
        let cut = faer::col::from_slice(s_image).norm_l1();
        let improved = if cut > self.c {
            self.c = cut;
            true
        } else {
            false
        };
        let sa = &mut self.s[axis];
        s_image.chunks(64).zip(sa).for_each(|(si, sa)| {
            let mut signs = 0u64;
            for (idx, &si) in si.iter().enumerate() {
                signs |= (si.is_sign_negative() as u64) << idx;
            }
            *sa = signs;
        });
        improved
    }

    fn blowup_along(
        dst: &mut [u64],
        axis: usize,
        dim: &[usize],
        s: &[Box<[u64]>],
        stack: PodStack<'_>,
    ) {
        let total_dim = dim.iter().product::<usize>();
        let full_len = total_dim / dim[axis];
        let limb_len = full_len.div_ceil(64);

        let kron = dst;
        {
            let (tmp, _) = stack.make_raw::<u64>(limb_len);
            kron[0] = 0;

            let mut len = 1usize;
            for (a, (&dim, s)) in iter::zip(dim, s).enumerate().rev() {
                if a != axis {
                    bit_kron(tmp, kron, len, s, dim);
                    len *= dim;
                    kron[..len.div_ceil(64)].copy_from_slice(&tmp[..len.div_ceil(64)]);
                }
            }
        }
    }

    pub fn blowup_mul_add(&self, dst: &mut [f32], scale: f32, stack: PodStack<'_>) {
        let full_len = self.total_dim;
        let limb_len = full_len.div_ceil(64);

        let (kron, stack) = stack.make_raw::<u64>(limb_len);
        {
            let (tmp, _) = stack.make_raw::<u64>(limb_len);
            kron[0] = 0;

            let mut len = 1usize;
            for (&dim, s) in iter::zip(&self.dim, &self.s).rev() {
                bit_kron(tmp, kron, len, s, dim);
                len *= dim;
                kron[..len.div_ceil(64)].copy_from_slice(&tmp[..len.div_ceil(64)]);
            }
        }

        for i in 0..full_len {
            let sign = ((kron[i / 64] >> (i % 64)) as u32) << 31;
            let value = f32::from_bits(scale.to_bits() ^ sign);
            dst[i] += value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyn_stack::{GlobalPodBuffer, StackReq};
    use half::bf16;
    use rand::prelude::*;

    fn expected_width(target: f32, curr: f32, w: usize) -> f32 {
        (w as f32) * target.ln() / curr.ln()
    }

    #[test]
    fn test_sct_tensor() {
        let dim = &*vec![64; 4];
        let order = dim.len();
        let stride = &mut *vec![0; order];
        {
            let mut s = 1usize;
            for (stride, dim) in iter::zip(&mut *stride, dim) {
                *stride = s;
                s *= *dim;
            }
        }
        let total_dim = dim.iter().product::<usize>();

        let rng = &mut StdRng::seed_from_u64(234324);
        let tensor = &*(0..total_dim)
            .map(|_| rand_distr::StandardNormal.sample(rng))
            .collect::<Box<[_]>>();

        let mut mem = {
            let sign = StackReq::new::<u64>(total_dim.div_ceil(64));
            let sign_f32 = StackReq::new::<f32>(total_dim);
            GlobalPodBuffer::new(sign.and(sign_f32))
        };
        let mut stack = PodStack::new(&mut mem);

        let mut remainder = Remainder::new(tensor, &dim, &stride);
        let init_norm = faer::col::from_slice(&remainder.t).norm_l2();

        let target_error = faer::col::from_slice(
            &tensor
                .iter()
                .map(|&x| (x - bf16::from_f32(x).to_f32()))
                .collect::<Box<[_]>>(),
        )
        .norm_l2()
            / init_norm;

        let width = 30;

        let target_bits = (total_dim * 16) as f32;
        let width_bits = remainder.width_bits();

        let mut cut = Cut::new(
            dim,
            remainder.two_mats.iter().map(|x| x.as_ref()),
            stack.rb_mut(),
        );

        let now = std::time::Instant::now();
        for w in 0..width {
            remainder.fill_matrices();
            remainder.cut(&mut cut, rng, stack.rb_mut());
            {
                let mut blowup = faer::Col::<f32>::zeros(total_dim);
                cut.blowup_mul_add(blowup.as_slice_mut(), 1.0, stack.rb_mut());

                let dot = faer::row::from_slice(&remainder.t) * blowup;
                let err = (dot - cut.c).abs() / f32::max(cut.c.abs(), dot.abs());
                dbg!(w, err);
            }
            remainder.update(&cut, stack.rb_mut());
            {
                let curr_error = remainder.norm_l2() / init_norm;
                let expected_width = expected_width(target_error, curr_error, w + 1);
                let expected_bits = expected_width * width_bits as f32;
                let expected_over_target = expected_bits / target_bits;
                dbg!(curr_error, target_error, expected_over_target,);
            }
        }
        dbg!(now.elapsed());
    }
}

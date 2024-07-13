use aligned_vec::ABox;
use core::{cell::Cell, iter};
use dyn_stack::PodStack;
use equator::assert;
use rand::Rng;
use reborrow::*;

pub struct Remainder {
    dim: Box<[usize]>,
    total_dim: usize,
    t: ABox<[f32]>,
    mats: Box<[faer::Mat<f32>]>,
}

struct SubIndices {
    basis: Box<[usize]>,
    dim: Box<[usize]>,
}

impl SubIndices {
    pub fn new(i: usize, dim: &[usize]) -> Self {
        let mut i = i;
        let order = dim.len();
        let mut basis = vec![0usize; order].into_boxed_slice();

        for axis in 0..order {
            let dim = dim[axis];
            basis[axis] = i % dim;
            i /= dim;
        }
        Self {
            basis,
            dim: dim.to_vec().into_boxed_slice(),
        }
    }

    pub fn entry(&self, axis: usize) -> (usize, usize) {
        let row = self.basis[axis];
        let mut col = 0;
        let mut stride = 1usize;
        for (a, (&basis, &dim)) in iter::zip(&self.basis, &self.dim).enumerate() {
            if a != axis {
                col += basis * stride;
                stride *= dim;
            }
        }
        (row, col)
    }
}

impl Remainder {
    pub fn new(tensor: &[f32], dim: &[usize], stride: &[usize]) -> Self {
        let order = dim.len();
        let total_dim = dim.iter().product::<usize>();
        assert!(total_dim != 0);

        let mut t = aligned_vec::avec![0.0f32; total_dim].into_boxed_slice();
        let mut state = vec![0usize; order].into_boxed_slice();
        let mut offset = vec![0usize; order].into_boxed_slice();

        let mut dst_index = 0usize;
        'outer: loop {
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

            let mut src_index = 0usize;
            for &offset in &*offset {
                src_index += offset;
            }

            t[dst_index] = tensor[src_index];

            dst_index += 1;
        }

        Self {
            dim: dim.to_vec().into_boxed_slice(),
            total_dim,
            t,
            mats: dim
                .iter()
                .map(|&dim| faer::Mat::zeros(dim, total_dim / dim))
                .collect(),
        }
    }

    pub fn norm_l2(&self) -> f32 {
        faer::col::from_slice(&self.t).norm_l2()
    }

    pub fn fill_matrices(&mut self) {
        for i in 0..self.total_dim {
            let ti = self.t[i];
            let i = SubIndices::new(i, &self.dim);
            for (axis, mat) in self.mats.iter_mut().enumerate() {
                let (row, col) = i.entry(axis);
                mat[(row, col)] = ti;
            }
        }
    }

    pub fn cut(&self, cut: &mut Cut, rng: &mut impl Rng, stack: PodStack<'_>) {
        let mut stack = stack;
        cut.setup(rng);
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
    c: f32,
    s: Box<[(usize, Box<[u64]>)]>,
    total_dim: usize,
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
    pub fn new(dim: &[usize]) -> Self {
        Self {
            c: f32::NEG_INFINITY,
            s: dim
                .iter()
                .map(|&dim| (dim, vec![0u64; dim.div_ceil(64)].into_boxed_slice()))
                .collect(),
            total_dim: dim.iter().product(),
        }
    }

    pub fn setup(&mut self, rng: &mut impl Rng) {
        self.c = f32::NEG_INFINITY;
        self.s
            .iter_mut()
            .flat_map(|(_, s)| s)
            .for_each(|s| *s = rng.gen())
    }

    pub fn dim(&self, axis: usize) -> usize {
        self.s[axis].0
    }

    pub fn improve(&mut self, axis: usize, stuff: &Remainder, stack: PodStack<'_>) -> bool {
        let (s, stack) = stack.make_raw::<f32>(self.total_dim / self.dim(axis));
        self.blowup_along(s, axis, stack);

        let s_image = &stuff.mats[axis] * faer::col::from_slice(&s);
        let cut = s_image.norm_l1();
        let improved = if cut > self.c {
            self.c = cut;
            true
        } else {
            false
        };
        let sa = &mut self.s[axis].1;
        s_image.as_slice().chunks(64).zip(sa).for_each(|(si, sa)| {
            let mut signs = 0u64;
            for (idx, &si) in si.iter().enumerate() {
                signs |= (si.is_sign_negative() as u64) << idx;
            }
            *sa = signs;
        });
        improved
    }

    pub fn blowup_along(&self, dst: &mut [f32], axis: usize, stack: PodStack<'_>) {
        let full_len = self.total_dim / self.dim(axis);
        let limb_len = full_len.div_ceil(64);

        let (kron, stack) = stack.make_raw::<u64>(limb_len);
        {
            let (tmp, _) = stack.make_raw::<u64>(limb_len);
            kron[0] = 0;

            let mut len = 1usize;
            for (a, s) in self.s.iter().enumerate().rev() {
                if a != axis {
                    bit_kron(tmp, kron, len, &s.1, s.0);
                    len *= s.0;
                    kron[..len.div_ceil(64)].copy_from_slice(&tmp[..len.div_ceil(64)]);
                }
            }
        }

        for i in 0..full_len {
            let signed_zero = f32::from_bits(((kron[i / 64] >> (i % 64)) as u32) << 31);
            dst[i] = f32::copysign(1.0, signed_zero);
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
            for s in self.s.iter().rev() {
                bit_kron(tmp, kron, len, &s.1, s.0);
                len *= s.0;
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
        let order = 3;
        let dim = &*vec![8; order];
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

        let target_error = faer::col::from_slice(
            &tensor
                .iter()
                .map(|&x| (x - bf16::from_f32(x).to_f32()))
                .collect::<Box<[_]>>(),
        )
        .norm_l2();

        let mut stuff = Remainder::new(tensor, &dim, &stride);
        let init_norm = faer::col::from_slice(&stuff.t).norm_l2();

        let width = 1000;

        let target_bits = (total_dim * 16) as f32;
        let width_bits = stuff.width_bits();

        let mut cut = Cut::new(dim);

        let mut mem = {
            let sign = StackReq::new::<u64>(total_dim.div_ceil(64));
            let sign_f32 = StackReq::new::<f32>(total_dim);
            GlobalPodBuffer::new(sign.and(sign_f32))
        };
        let mut stack = PodStack::new(&mut mem);

        for w in 0..width {
            stuff.fill_matrices();
            stuff.cut(&mut cut, rng, stack.rb_mut());
            {
                let mut blowup = faer::Col::<f32>::zeros(total_dim);
                cut.blowup_mul_add(blowup.as_slice_mut(), 1.0, stack.rb_mut());

                let dot = faer::row::from_slice(&stuff.t) * blowup;
                let err = (dot - cut.c).abs() / f32::max(cut.c.abs(), dot.abs());
                dbg!(w, err);
            }
            stuff.update(&cut, stack.rb_mut());
            {
                let curr_error = faer::col::from_slice(&stuff.t).norm_l2() / init_norm;
                let expected_width = expected_width(target_error, curr_error, w + 1);
                let expected_bits = expected_width * width_bits as f32;
                let expected_over_target = expected_bits / target_bits;
                dbg!(curr_error, target_error, expected_over_target,);
            }
        }
    }
}

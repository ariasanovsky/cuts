use aligned_vec::{avec, ABox};
use core::{cell::Cell, iter};
use dyn_stack::PodStack;
use equator::assert;
use rand::Rng;
use reborrow::*;

use crate::{
    bitmagic,
    inplace_sct::{sparse_matvec, sparse_matvec_sign},
    SignMatRef,
};

// copied from the standard library since `usize::isqrt` is currently unstable.
fn isqrt(n: usize) -> usize {
    if n < 2 {
        return n;
    }

    let mut op = n;
    let mut res = 0;
    let mut one = 1 << (n.ilog2() & !1);

    while one != 0 {
        if op >= res + one {
            op -= res + one;
            res = (res >> 1) + one;
        } else {
            res >>= 1;
        }
        one >>= 2;
    }

    res
}

#[track_caller]
fn sparse_update(
    matvec: &mut [f32],
    mat: faer::MatRef<'_, f32>,
    s: SignMatRef<'_>,
    t: SignMatRef<'_>,
    c: &[f32],
    scale: f32,

    x_new: &[u64],
    x_old: &[u64],
    stack: PodStack<'_>,
) {
    let n = t.nrows();
    let (diff_indices, _) = stack.make_raw::<usize>(n);

    let mut pos = 0usize;
    let mut pos_rev = n;

    {
        let x_new: &[u8] = bytemuck::cast_slice(x_new);
        let x_old: &[u8] = bytemuck::cast_slice(x_old);
        for j in 0..n.div_ceil(8) {
            let x_new = x_new[j];
            let x_old = x_old[j];

            let j = j * 8;

            for i in 0..8 {
                let x_neg = (x_new >> i) & 1 == 1;
                let x_neg_old = (x_old >> i) & 1 == 1;

                if x_neg != x_neg_old {
                    if x_neg {
                        diff_indices[pos] = j + i;
                        pos += 1;
                    } else {
                        pos_rev -= 1;
                        diff_indices[pos_rev] = j + i;
                    }
                }
            }
        }
    }

    let (diff_indices, diff_indices_pos) = diff_indices.split_at(pos_rev);
    let diff_indices_neg = &diff_indices[..pos];
    {
        sparse_matvec_sign(matvec, mat, diff_indices_neg, true);
        sparse_matvec_sign(matvec, mat, diff_indices_pos, false);
    }

    {
        let f = scale;
        let y = &mut *avec![0.0_f32; c.len()].into_boxed_slice();

        if diff_indices_neg.len() + diff_indices_pos.len() < n / 8 {
            for (idx, (y, &c)) in iter::zip(&mut *y, c).enumerate() {
                let t = t.storage().col_as_slice(idx);

                let dot = {
                    let mut pos = 0u64;
                    let mut neg = 0u64;

                    for &i in diff_indices_pos {
                        pos += (t[i / 64] >> (i % 64)) & 1;
                    }
                    for &i in diff_indices_neg {
                        neg += (t[i / 64] >> (i % 64)) & 1;
                    }

                    let pos = (diff_indices_pos.len() as i64) - 2 * pos as i64;
                    let neg = (diff_indices_neg.len() as i64) - 2 * neg as i64;

                    pos - neg
                };
                *y = (dot as f32) * (c * f);
            }
        } else {
            for (idx, (y, &c)) in iter::zip(&mut *y, c).enumerate() {
                let t = t.storage().col_as_slice(idx);

                let dot = {
                    let mut old_acc = 0u64;
                    let mut new_acc = 0u64;

                    for (&t, (&old, &new)) in iter::zip(t, iter::zip(x_old, x_new)) {
                        old_acc += u64::count_ones(t ^ old) as u64;
                        new_acc += u64::count_ones(t ^ new) as u64;
                    }

                    old_acc as i64 - new_acc as i64
                };
                *y = (dot as f32) * (c * f);
            }
        }

        bitmagic::matvec::matvec_f32(matvec, s, y);
    }
}

fn partition<T>(slice: &[T], n_partitions: usize) -> impl Iterator<Item = &[T]> {
    let mut slice = slice;

    let div = slice.len() / n_partitions;
    let mut rem = slice.len() % n_partitions;
    let mut count = n_partitions;

    iter::from_fn(move || {
        if count == 0 {
            return None;
        }
        count -= 1;
        let extra = (rem != 0) as usize;

        let next_len = div + extra;
        rem -= extra;

        let next;
        (next, slice) = core::mem::take(&mut slice).split_at(next_len);
        Some(next)
    })
}

fn partition_mut<T>(slice: &mut [T], n_partitions: usize) -> impl Iterator<Item = &mut [T]> {
    let mut slice = slice;

    let div = slice.len() / n_partitions;
    let mut rem = slice.len() % n_partitions;
    let mut count = n_partitions;

    iter::from_fn(move || {
        if count == 0 {
            return None;
        }
        count -= 1;
        let extra = (rem != 0) as usize;

        let next_len = div + extra;
        rem -= extra;

        let next;
        (next, slice) = core::mem::take(&mut slice).split_at_mut(next_len);
        Some(next)
    })
}

#[derive(Clone, Debug)]
pub struct Remainder {
    dim: Box<[usize]>,
    total_dim: usize,
    t: ABox<[f32]>,
    mats: Box<[faer::Mat<f32>]>,
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
            let t = &mut self.mats[axis];
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

                if axis_stride == 1 {
                    t.col_as_slice_mut(dst_col)
                        .copy_from_slice(&tensor[src_index..][..dim[axis]]);
                } else {
                    let t = t.as_mut().col_mut(dst_col);
                    let tensor = &tensor[src_index..];

                    for (dst, &src) in iter::zip(t.iter_mut(), tensor.iter().step_by(axis_stride)) {
                        *dst = src;
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

    pub fn cut(&self, cut: &mut Cut, rng: &mut impl Rng, stack: PodStack<'_>) {
        let mut stack = stack;
        cut.setup(self, rng, stack.rb_mut());
        loop {
            let mut improved = false;
            for axis in 0..self.dim.len() {
                improved |= cut.improve(axis, self, stack.rb_mut());
            }
            if !improved {
                cut.how_full += 1;
                return;
            }
        }
    }

    pub fn update(&mut self, cut: &Cut, stack: PodStack<'_>) {
        let f = -1.0 / self.total_dim as f32;
        let scale = &*cut.c().iter().map(|&c| c * f).collect::<Box<[_]>>();
        cut.flush(&mut self.t, scale, stack);
    }

    pub fn width_bits(&self) -> usize {
        32 + self.dim.iter().sum::<usize>()
    }

    pub fn mats(&self) -> impl IntoIterator<Item = faer::MatRef<f32>> {
        self.mats.iter().map(|mat| mat.as_ref())
    }

    pub fn t(&self) -> &[f32] {
        &self.t
    }
}

#[derive(Clone, Debug)]
pub struct Cut {
    dim: Box<[usize]>,
    s_axes: Box<[usize]>,
    t_axes: Box<[usize]>,

    s_dim: usize,
    t_dim: usize,
    total_dim: usize,

    blocksize: usize,
    how_full: usize,
    signs: Box<[Box<[u64]>]>,
    c: Box<[f32]>,
    x_signs: Box<[Box<[u64]>]>,
    signs_matvec: Box<[ABox<[f32]>]>,
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
    pub fn c(&self) -> &[f32] {
        &self.c[..self.how_full]
    }

    pub fn new<'a>(dim: &[usize], blocksize: usize) -> Self {
        let signs = dim
            .iter()
            .map(|&dim| vec![0u64; blocksize * dim.div_ceil(64)].into_boxed_slice())
            .collect::<Box<[_]>>();
        let total_dim = dim.iter().product::<usize>();

        let argsort = (0..dim.len()).collect::<Box<[_]>>();
        // argsort.sort_unstable_by_key(|&axis| dim[axis]);

        let mut cutoff = 0usize;
        let mut s_dim = 1usize;
        let total_dim_sqrt = isqrt(total_dim);
        while s_dim < total_dim_sqrt {
            s_dim *= dim[argsort[cutoff]];
            cutoff += 1;
        }

        let x_signs = dim
            .iter()
            .enumerate()
            .map(|(axis, _)| {
                let n = total_dim / dim[axis];
                vec![0u64; n.div_ceil(64) * blocksize].into_boxed_slice()
            })
            .collect();

        Self {
            dim: dim.to_vec().into_boxed_slice(),
            s_axes: argsort[..cutoff].to_vec().into_boxed_slice(),
            t_axes: argsort[cutoff..].to_vec().into_boxed_slice(),

            s_dim,
            t_dim: total_dim / s_dim,
            total_dim,

            blocksize,
            how_full: 0,
            signs,
            x_signs,

            signs_matvec: dim
                .iter()
                .map(|&dim| avec![0.0_f32; dim].into_boxed_slice())
                .collect(),
            c: vec![f32::NEG_INFINITY; blocksize].into_boxed_slice(),
        }
    }

    fn signs_at<'a, I, Item: 'a + ?Sized>(
        signs: I,
        blocksize: usize,
        block_idx: usize,
    ) -> impl Clone + ExactSizeIterator + DoubleEndedIterator<Item = &'a [u64]>
    where
        I: IntoIterator<Item = &'a Item>,
        Item: AsRef<[u64]>,
        I::IntoIter: Clone + ExactSizeIterator + DoubleEndedIterator,
    {
        assert!(block_idx < blocksize);
        signs
            .into_iter()
            .map(move |signs| partition(signs.as_ref(), blocksize).nth(block_idx).unwrap())
    }

    fn signs_at_mut<'a, I, Item: 'a + ?Sized>(
        signs: I,
        blocksize: usize,
        block_idx: usize,
    ) -> impl ExactSizeIterator + DoubleEndedIterator<Item = &'a mut [u64]>
    where
        I: IntoIterator<Item = &'a mut Item>,
        Item: AsMut<[u64]>,
        I::IntoIter: ExactSizeIterator + DoubleEndedIterator,
    {
        assert!(block_idx < blocksize);
        signs.into_iter().map(move |signs| {
            partition_mut(signs.as_mut(), blocksize)
                .nth(block_idx)
                .unwrap()
        })
    }

    #[inline(never)]
    fn update_image(&mut self, axis: usize, remainder: &Remainder, stack: PodStack<'_>) {
        let how_full = self.how_full;
        let blocksize = self.blocksize;

        let matvec = &mut *self.signs_matvec[axis];

        let mut stack = stack;
        let n = self.total_dim / self.dim[axis];
        let (x_new, mut stack) = stack.rb_mut().make_raw::<u64>(n.div_ceil(64));

        Self::blowup_along(
            x_new,
            axis,
            &self.dim,
            Self::signs_at(&self.signs, blocksize, how_full),
            stack.rb_mut(),
        );

        let x_new = &*x_new;
        let x_old = Self::signs_at(&self.x_signs, blocksize, how_full)
            .nth(axis)
            .unwrap();

        {
            let s = &*self.signs[axis];
            let dim = self.dim[axis];
            let len = dim.div_ceil(64);
            let s = &s[..len * how_full];
            let s = SignMatRef::from_storage(
                crate::MatRef::from_col_major_slice(s, len, how_full, len),
                dim,
            );

            let t = &*self.x_signs[axis];
            let dim = n;
            let len = dim.div_ceil(64);
            let t = &t[..len * how_full];
            let t = SignMatRef::from_storage(
                crate::MatRef::from_col_major_slice(t, len, how_full, len),
                dim,
            );

            sparse_update(
                matvec,
                remainder.mats[axis].as_ref(),
                s,
                t,
                &self.c[..how_full],
                -1.0 / self.total_dim as f32,
                x_new,
                x_old,
                stack.rb_mut(),
            );
        }

        let x_old = Self::signs_at_mut(&mut self.x_signs, blocksize, how_full)
            .nth(axis)
            .unwrap();
        x_old.copy_from_slice(x_new);
    }

    #[inline(never)]
    pub fn setup(&mut self, remainder: &Remainder, rng: &mut impl Rng, stack: PodStack<'_>) {
        let how_full = self.how_full;
        let blocksize = self.blocksize;
        assert!(how_full < blocksize);
        self.c[how_full] = f32::NEG_INFINITY;
        let mut stack = stack;

        Self::signs_at_mut(&mut self.signs, blocksize, how_full)
            .flatten()
            .for_each(|s| *s = rng.gen());

        for axis in 0..self.dim.len() {
            let n = self.total_dim / self.dim[axis];
            let matvec = &mut *self.signs_matvec[axis];

            let x_new = Self::signs_at_mut(&mut self.x_signs, blocksize, how_full)
                .nth(axis)
                .unwrap();
            Self::blowup_along(
                x_new,
                axis,
                &self.dim,
                Self::signs_at(&self.signs, blocksize, how_full),
                stack.rb_mut(),
            );
            let x_new = Self::signs_at(&self.x_signs, blocksize, how_full)
                .nth(axis)
                .unwrap();

            if how_full > 0 {
                let prev_signs = Self::signs_at(&self.signs, blocksize, how_full - 1)
                    .nth(axis)
                    .unwrap();
                let c = 0.5 * self.c[how_full - 1] / self.dim[axis] as f32;

                for (x, &s) in iter::zip(matvec.chunks_mut(64), prev_signs) {
                    for (i, x) in x.iter_mut().enumerate() {
                        let sign = ((s >> i) as u32) << 31;
                        *x -= f32::from_bits(c.to_bits() ^ sign)
                    }
                }

                let x_old = Self::signs_at(&self.x_signs, blocksize, how_full - 1)
                    .nth(axis)
                    .unwrap();

                let s = &*self.signs[axis];
                let dim = self.dim[axis];
                let len = dim.div_ceil(64);
                let s = &s[..len * how_full];
                let s = SignMatRef::from_storage(
                    crate::MatRef::from_col_major_slice(s, len, how_full, len),
                    dim,
                );

                let t = &*self.x_signs[axis];
                let dim = n;
                let len = dim.div_ceil(64);
                let t = &t[..len * how_full];
                let t = SignMatRef::from_storage(
                    crate::MatRef::from_col_major_slice(t, len, how_full, len),
                    dim,
                );

                sparse_update(
                    matvec,
                    remainder.mats[axis].as_ref(),
                    s,
                    t,
                    &self.c[..how_full],
                    -1.0 / self.total_dim as f32,
                    x_new,
                    x_old,
                    stack.rb_mut(),
                );
            } else {
                let (diff_indices, _) = stack.rb_mut().make_raw::<u64>(n);
                let mut pos = 0usize;
                for j in 0..n {
                    let s_neg = (x_new[j / 64] >> (j % 64)) & 1 == 1;
                    diff_indices[pos] = ((s_neg as u64) << 63) | j as u64;
                    pos += 1;
                }
                let diff_indices = &diff_indices[..pos];

                matvec.fill(0.0);
                sparse_matvec(matvec, remainder.mats[axis].as_ref(), diff_indices);

                for x in matvec {
                    *x = *x * 0.5;
                }
            }
        }
    }

    pub fn dim(&self, axis: usize) -> usize {
        self.dim[axis]
    }

    #[inline(never)]
    pub fn improve(&mut self, axis: usize, remainder: &Remainder, stack: PodStack<'_>) -> bool {
        let mut stack = stack;
        self.update_image(axis, remainder, stack.rb_mut());

        let s_image = &*self.signs_matvec[axis];
        let cut = faer::col::from_slice(s_image).norm_l1() * 2.0;
        let improved = if cut > self.c[self.how_full] {
            self.c[self.how_full] = cut;
            true
        } else {
            false
        };

        let signs = Self::signs_at_mut(&mut self.signs, self.blocksize, self.how_full)
            .nth(axis)
            .unwrap();
        iter::zip(s_image.chunks(64), signs).for_each(|(si, sa)| {
            let mut signs = 0u64;
            for (idx, &si) in si.iter().enumerate() {
                signs |= (si.is_sign_negative() as u64) << idx;
            }
            *sa = signs;
        });
        improved
    }

    #[inline(never)]
    fn blowup_along<'a>(
        dst: &mut [u64],
        axis: usize,
        dim: &[usize],
        s: impl Clone + DoubleEndedIterator<Item = &'a [u64]> + ExactSizeIterator,
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
            for (a, (&dim, s)) in iter::zip(dim, s.clone()).enumerate().rev() {
                if a != axis {
                    bit_kron(tmp, kron, len, s, dim);
                    len *= dim;
                    kron[..len.div_ceil(64)].copy_from_slice(&tmp[..len.div_ceil(64)]);
                }
            }
        }
    }

    pub fn flush(&self, dst: &mut [f32], scale: &[f32], stack: PodStack<'_>) {
        let how_full = self.how_full;
        let blocksize = self.blocksize;
        assert!(how_full <= blocksize);

        let rank = self.how_full;

        let s_len = self.s_dim;
        let t_len = self.t_dim;
        let s_limb_len = s_len.div_ceil(64);
        let t_limb_len = t_len.div_ceil(64);

        let (s_kron, stack) = stack.make_raw::<u64>(s_limb_len * rank);
        let (t_kron, mut stack) = stack.make_raw::<u64>(t_limb_len * rank);

        for (idx, (s_kron, t_kron)) in
            iter::zip(partition_mut(s_kron, rank), partition_mut(t_kron, rank)).enumerate()
        {
            for (dst, ax) in [(s_kron, &*self.s_axes), (t_kron, &*self.t_axes)] {
                let (tmp, _) = stack.rb_mut().make_raw::<u64>(dst.len());
                dst[0] = 0;

                let mut len = 1usize;
                for &a in ax.iter().rev() {
                    let s = Self::signs_at(&self.signs, self.blocksize, idx)
                        .nth(a)
                        .unwrap();
                    let dim = self.dim[a];

                    bit_kron(tmp, dst, len, s, dim);
                    len *= dim;
                    dst[..len.div_ceil(64)].copy_from_slice(&tmp[..len.div_ceil(64)]);
                }
            }
        }

        let s = SignMatRef::from_storage(
            crate::MatRef::from_col_major_slice(s_kron, s_limb_len, rank, s_limb_len),
            s_len,
        );
        let t = SignMatRef::from_storage(
            crate::MatRef::from_col_major_slice(t_kron, t_limb_len, rank, t_limb_len),
            t_len,
        );
        bitmagic::matmul::mat_tmat_f32(
            crate::MatMut::from_faer(faer::mat::from_column_major_slice_mut(dst, s_len, t_len)),
            s,
            t,
            scale,
        );
    }

    pub fn reset(&mut self) {
        self.how_full = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dyn_stack::{GlobalPodBuffer, StackReq};
    use equator::assert;
    use half::bf16;
    use rand::prelude::*;

    #[test]
    fn test_sct_tensor() {
        let dim = &*vec![17, 97, 127];
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
        let blocksize = 32;

        let rng = &mut StdRng::seed_from_u64(234324);
        let tensor = &*(0..total_dim)
            .map(|_| rand_distr::StandardNormal.sample(rng))
            .collect::<Box<[_]>>();

        let mut remainder = Remainder::new(tensor, &dim, &stride);
        let init_norm = faer::col::from_slice(&remainder.t).norm_l2();

        let mut cut = Cut::new(dim, blocksize);

        let mut mem = {
            let col_dim = dim
                .iter()
                .map(|&dim| cut.total_dim / dim)
                .max()
                .unwrap_or(0);

            let x_new = StackReq::new::<u64>(col_dim.div_ceil(64));
            let diff_idx = StackReq::new::<usize>(col_dim);
            let kron = StackReq::new::<u64>(cut.total_dim.div_ceil(64));
            let s_kron = StackReq::new::<u64>(cut.s_dim.div_ceil(64) * blocksize);
            let t_kron = StackReq::new::<u64>(cut.t_dim.div_ceil(64) * blocksize);
            GlobalPodBuffer::new(StackReq::any_of([
                StackReq::all_of([x_new, diff_idx]),
                StackReq::all_of([s_kron.or(t_kron), s_kron, t_kron]),
                StackReq::all_of([kron, kron]),
            ]))
        };
        let mut stack = PodStack::new(&mut mem);

        let target_error = faer::col::from_slice(
            &tensor
                .iter()
                .map(|&x| (x - bf16::from_f32(x).to_f32()))
                .collect::<Box<[_]>>(),
        )
        .norm_l2()
            / init_norm;

        let width = usize::MAX;
        let mut w = 0;
        while w < width {
            remainder.fill_matrices();
            let bs = Ord::min(blocksize, width - w);
            for _ in 0..bs {
                remainder.cut(&mut cut, rng, stack.rb_mut());
            }
            remainder.update(&cut, stack.rb_mut());
            cut.reset();
            if remainder.norm_l2() / init_norm < target_error {
                break;
            }
            w += bs;
        }
        assert!(remainder.norm_l2() / init_norm < target_error);
    }
}

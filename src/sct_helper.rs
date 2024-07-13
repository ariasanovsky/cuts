use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use faer::{linalg::temp_mat_req, Mat, MatRef};
use rand::Rng;
use reborrow::{Reborrow, ReborrowMut};

use crate::{bitmagic::matmul::mat_tmat_f32, inplace_sct::CutHelper, sct::{Sct, SctMut, SctRef}};

pub fn width(nrows: usize, ncols: usize, compression_rate: f64) -> usize {
    let numerator = compression_rate * (nrows * ncols) as f64;
    let denominator = (4 + nrows.next_multiple_of(64) / 8 + ncols.next_multiple_of(64)) as f64;
    let width = (numerator / denominator) as usize;
    Ord::max(8, width)
}

pub struct SctHelper {
    block: Sct,
    sct: Sct,
    nrows: usize,
    ncols: usize,
    block_size: usize,
    how_full: usize,
    mem: GlobalPodBuffer,
    remainder_norm: f32,
    two_remainder: Mat<f32>,
    two_remainder_transposed: Mat<f32>,
    cut_helper: CutHelper,
}

impl SctHelper {
    pub fn new(mat: MatRef<f32>, block_size: usize, width: usize) -> Self {
        let (nrows, ncols) = mat.shape();
        let block = Sct::new(nrows, ncols, block_size);
        let sct = Sct::new(nrows, ncols, width);
        let mem = GlobalPodBuffer::new(
            StackReq::new::<u64>(Ord::max(nrows, ncols))
                .and(temp_mat_req::<f32>(block_size, 1).unwrap()),
        );
        let remainder_norm = mat.squared_norm_l2();
        let two_remainder = faer::scale(2.0f32) * &mat;
        let two_remainder_transposed = faer::scale(2.0f32) * mat.transpose();
        let cut_helper = CutHelper::new(two_remainder.as_ref(), two_remainder_transposed.as_ref());
        Self {
            block,
            sct,
            nrows,
            ncols,
            block_size,
            how_full: 0,
            mem,
            remainder_norm,
            two_remainder,
            two_remainder_transposed,
            cut_helper,
        }
    }

    pub fn cut(&mut self, rng: &mut impl Rng) -> f32 {
        if self.how_full == self.block_size {
            self.flush();
        }
        let Self {
            block,
            sct: _,
            nrows,
            ncols,
            block_size: _,
            how_full,
            mem,
            remainder_norm,
            two_remainder,
            two_remainder_transposed,
            cut_helper,
        } = self;
        *how_full += 1;
        let SctMut { mut s, c, mut t } = block.as_mut();
        let mut stack = PodStack::new(mem);
        let cut = cut_helper.cut_mat(
            two_remainder.as_ref(),
            two_remainder_transposed.as_ref(),
            s.rb_mut().split_at_col_mut(*how_full).0,
            faer::col::from_slice_mut(&mut c[..*how_full]),
            t.rb_mut().split_at_col_mut(*how_full).0,
            rng,
            usize::MAX,
            stack.rb_mut(),
        );
        *remainder_norm -= (cut * cut) / (*nrows * *ncols) as f32;
        cut
    }

    pub fn flush(&mut self) {
        let Self {
            block,
            sct,
            nrows,
            ncols,
            block_size: _,
            how_full,
            mem: _,
            remainder_norm,
            two_remainder,
            two_remainder_transposed,
            cut_helper: _,
        } = self;
        {
            let SctMut { mut s, c, mut t } = block.as_mut();
            let two_remainder =
                crate::MatMut::from_faer(two_remainder.as_mut());
            crate::bitmagic::matmul::mat_tmat_f32(
                two_remainder,
                s.rb_mut().split_at_col_mut(*how_full).0.rb(),
                t.rb_mut().split_at_col_mut(*how_full).0.rb(),
                &c[..*how_full],
            );
        }
        two_remainder_transposed
            .as_mut()
            .copy_from(two_remainder.transpose());
        *remainder_norm = two_remainder.squared_norm_l2() / 4.0;

        let SctRef { s, c, t } = block.as_ref();
        for k in 0..*how_full {
            sct.s[nrows.div_ceil(64) * sct.how_full..]
                [..nrows.div_ceil(64)]
                .copy_from_slice(s.rb().storage().col_as_slice(k));
            sct.t[ncols.div_ceil(64) * sct.how_full..]
                [..ncols.div_ceil(64)]
                .copy_from_slice(t.rb().storage().col_as_slice(k));
            sct.c[sct.how_full] = -c[k] / 2.0;
            sct.how_full += 1;
        }
        *how_full = 0;
    }

    pub fn squared_norm_l2(&self) -> f32 {
        self.remainder_norm
    }

    pub fn expand(&self) -> Mat<f32> {
        let mut mat = Mat::zeros(self.nrows, self.ncols);
        let mat_mut = crate::MatMut::from_faer(mat.as_mut());
        let SctRef { s, c, t } = self.sct.as_ref();
        mat_tmat_f32(mat_mut, s, t, c);
        mat
    }
}

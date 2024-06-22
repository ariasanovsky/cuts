use faer::Col;
use safetensors::View;

use crate::{SignMatMut, SignMatRef};

pub struct Sct {
    pub s: Vec<u64>,
    pub c: Col<f32>,
    pub t: Vec<u64>,
    pub nrows: usize,
    pub ncols: usize,
    pub width: usize,
    pub how_full: usize,
}

impl Sct {
    pub fn new(nrows: usize, ncols: usize, width: usize) -> Self {
        Self {
            s: vec![0u64; nrows.div_ceil(64) * width],
            c: Col::zeros(width),
            t: vec![0u64; ncols.div_ceil(64) * width],
            nrows,
            ncols,
            width,
            how_full: 0,
        }
    }

    pub fn as_ref(&self) -> SctRef {
        let Self { s, c, t, nrows, ncols, width, how_full } = self;
        SctRef {
            s: SignMatRef::from_storage(
                crate::MatRef::from_col_major_slice(
                    s,
                    nrows.div_ceil(64),
                    *width,
                    nrows.div_ceil(64),
                ),
                *nrows
            ),
            c,
            t: SignMatRef::from_storage(
                crate::MatRef::from_col_major_slice(
                    t,
                    ncols.div_ceil(64),
                    *width,
                    ncols.div_ceil(64),
                ),
                *ncols,
            ),
        }
    }

    pub fn as_mut(&mut self) -> SctMut {
        let Self { s, c, t, nrows, ncols, width, how_full } = self;
        SctMut {
            s: SignMatMut::from_storage(
                crate::MatMut::from_col_major_slice(
                    s,
                    nrows.div_ceil(64),
                    *width,
                    nrows.div_ceil(64),
                ),
                *nrows
            ),
            c,
            t: SignMatMut::from_storage(
                crate::MatMut::from_col_major_slice(
                    t,
                    ncols.div_ceil(64),
                    *width,
                    ncols.div_ceil(64),
                ),
                *ncols,
            ),
        }
    }
}

pub struct SctRef<'a> {
    pub s: SignMatRef<'a>,
    pub c: &'a Col<f32>,
    pub t: SignMatRef<'a>,
}

pub struct SctMut<'a> {
    pub s: SignMatMut<'a>,
    pub c: &'a mut Col<f32>,
    pub t: SignMatMut<'a>,
}

impl Sct {
    pub fn views(&self) -> impl IntoIterator<Item = (String, impl View + '_)> {
        let Self { s, c, t, nrows, ncols, width, how_full } = self;
        struct DynView<'a>(Box<dyn View + 'a>);
        impl<'a> View for DynView<'a> {
            fn dtype(&self) -> safetensors::Dtype {
                self.0.dtype()
            }
        
            fn shape(&self) -> &[usize] {
                self.0.shape()
            }
        
            fn data(&self) -> std::borrow::Cow<[u8]> {
                self.0.data()
            }
        
            fn data_len(&self) -> usize {
                self.0.data_len()
            }
        }
        struct Slice<'a, T>(&'a [T], usize);
        trait DType {
            fn dtype() -> safetensors::Dtype;
        }
        impl<'a, T: DType + bytemuck::Pod> View for Slice<'a, T> {
            fn dtype(&self) -> safetensors::Dtype {
                T::dtype()
            }
        
            fn shape(&self) -> &[usize] {
                std::slice::from_ref(&self.1)
            }
        
            fn data(&self) -> std::borrow::Cow<[u8]> {
                bytemuck::cast_slice(self.0).into()
            }
        
            fn data_len(&self) -> usize {
                self.data().len()
            }
        }
        impl DType for u64 {
            fn dtype() -> safetensors::Dtype {
                safetensors::Dtype::U64
            }
        }
        impl DType for f32 {
            fn dtype() -> safetensors::Dtype {
                safetensors::Dtype::F32
            }
        }
        [
            ("s".to_owned(), DynView(Box::new(Slice(s.as_slice(), s.len())))),
            ("t".to_owned(), DynView(Box::new(Slice(t.as_slice(), t.len())))),
            ("c".to_owned(), DynView(Box::new(Slice(c.as_slice(), c.nrows())))),
        ]
    }
}


#[cfg(test)]
mod tests {
    use crate::inplace_sct::CutHelper;

    use super::*;
    use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
    use faer::{linalg::temp_mat_req, Mat};
    use rand::{distributions::Distribution, rngs::StdRng, SeedableRng};
    use reborrow::{Reborrow, ReborrowMut};

    #[test]
    fn test_cuts_regular_sct() {
        let dim = 512;
        let nrows = dim;
        let ncols = dim;
        let blocksize = 32;

        let rng = &mut StdRng::seed_from_u64(0);

        let A: Mat<f32> = faer::stats::StandardNormalMat { nrows, ncols }.sample(rng);
        let init_norm = A.squared_norm_l2();

        let mut two_remainder: Mat<f32> = faer::scale(2.0f32) * &A;
        let mut two_remainder_transposed = faer::scale(2.0f32) * A.transpose();
        let mut helper = CutHelper::new(two_remainder.as_ref(), two_remainder_transposed.as_ref());
        let mut S = vec![0u64; (nrows.div_ceil(64)) * blocksize].into_boxed_slice();
        let mut T = vec![0u64; (ncols.div_ceil(64)) * blocksize].into_boxed_slice();
        let mut C = Col::<f32>::zeros(blocksize);

        let bf16_residual = Mat::from_fn(nrows, ncols, |i, j| {
            A[(i, j)] - half::bf16::from_f32(A[(i, j)]).to_f32()
        })
        .squared_norm_l2();

        let mut S = SignMatMut::from_storage(
            crate::MatMut::from_col_major_slice(
                &mut S,
                nrows.div_ceil(64),
                blocksize,
                nrows.div_ceil(64),
            ),
            nrows,
        );
        let mut T = SignMatMut::from_storage(
            crate::MatMut::from_col_major_slice(
                &mut T,
                ncols.div_ceil(64),
                blocksize,
                ncols.div_ceil(64),
            ),
            ncols,
        );
        let mut how_full = 0usize;
        let mut mem = GlobalPodBuffer::new(
            StackReq::new::<u64>(Ord::max(nrows, ncols))
                .and(temp_mat_req::<f32>(blocksize, 1).unwrap()),
        );
        let mut stack = PodStack::new(&mut mem);

        let mut full_S: Vec<u64> = vec![];
        let mut full_T: Vec<u64> = vec![];
        let mut full_C: Vec<f32> = vec![];

        let mut remainder_norm = init_norm;

        let mut iter = 0;
        while iter < 20000 {
            if how_full == blocksize {
                {
                    let two_remainder = crate::MatMut::from_faer(two_remainder.as_mut());
                    crate::bitmagic::matmul::mat_tmat_f32(
                        two_remainder,
                        S.rb(),
                        T.rb(),
                        C.as_slice(),
                    );
                }

                two_remainder_transposed
                    .as_mut()
                    .copy_from(two_remainder.transpose());
                remainder_norm = two_remainder.squared_norm_l2() / 4.0;

                for k in 0..blocksize {
                    full_S.extend_from_slice(S.rb().storage().col_as_slice(k));
                    full_T.extend_from_slice(T.rb().storage().col_as_slice(k));
                    full_C.push(-C[k] / 2.0);
                }
                how_full = 0;
            }

            how_full += 1;
            let cut = helper.cut_mat(
                two_remainder.as_ref(),
                two_remainder_transposed.as_ref(),
                S.rb_mut().split_at_col_mut(how_full).0,
                C.get_mut(..how_full),
                T.rb_mut().split_at_col_mut(how_full).0,
                rng,
                usize::MAX,
                stack.rb_mut(),
            );

            remainder_norm -= (cut * cut) / (nrows * ncols) as f32;
            if remainder_norm <= bf16_residual {
                {
                    let two_remainder = crate::MatMut::from_faer(two_remainder.as_mut());
                    crate::bitmagic::matmul::mat_tmat_f32(
                        two_remainder,
                        S.rb_mut().split_at_col_mut(how_full).0.rb(),
                        T.rb_mut().split_at_col_mut(how_full).0.rb(),
                        &C.as_slice()[..how_full],
                    );
                }
                two_remainder_transposed
                    .as_mut()
                    .copy_from(two_remainder.transpose());

                for k in 0..how_full {
                    full_S.extend_from_slice(S.rb().storage().col_as_slice(k));
                    full_T.extend_from_slice(T.rb().storage().col_as_slice(k));
                    full_C.push(-C[k] / 2.0);
                }

                break;
            }

            iter += 1;
        }
        dbg!(iter);
        dbg!(remainder_norm);

        let break_even_point = iter;

        let sct_bytes = ((nrows / 8 + ncols / 8 + 4) * break_even_point) as f64;
        let f32_bytes = (nrows * ncols * 4) as f64;
        let f16_bytes = (nrows * ncols * 2) as f64;

        eprintln!("dim: ({nrows}*{ncols}), break even at: {break_even_point}, f32 compression rate: {:.3}, bf16 compression rate: {:.3}", sct_bytes / f32_bytes, sct_bytes / f16_bytes);
    }
}
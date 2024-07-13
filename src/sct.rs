use faer::Col;
use safetensors::View;

use crate::{SignMatMut, SignMatRef};

pub struct Sct {
    pub s: Box<[u64]>,
    pub c: Col<f32>,
    pub t: Box<[u64]>,
    pub nrows: usize,
    pub ncols: usize,
    pub width: usize,
    pub how_full: usize,
}

impl Sct {
    pub fn new(nrows: usize, ncols: usize, width: usize) -> Self {
        Self {
            s: vec![0u64; nrows.div_ceil(64) * width].into_boxed_slice(),
            c: Col::zeros(width),
            t: vec![0u64; ncols.div_ceil(64) * width].into_boxed_slice(),
            nrows,
            ncols,
            width,
            how_full: 0,
        }
    }

    pub fn as_ref(&self) -> SctRef {
        let Self {
            s,
            c,
            t,
            nrows,
            ncols,
            width,
            ..
        } = self;
        SctRef {
            s: SignMatRef::from_storage(
                crate::MatRef::from_col_major_slice(
                    s,
                    nrows.div_ceil(64),
                    *width,
                    nrows.div_ceil(64),
                ),
                *nrows,
            ),
            c: c.as_slice(),
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
        let Self {
            s,
            c,
            t,
            nrows,
            ncols,
            width,
            ..
        } = self;
        SctMut {
            s: SignMatMut::from_storage(
                crate::MatMut::from_col_major_slice(
                    s,
                    nrows.div_ceil(64),
                    *width,
                    nrows.div_ceil(64),
                ),
                *nrows,
            ),
            c: c.as_slice_mut(),
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
    pub c: &'a [f32],
    pub t: SignMatRef<'a>,
}

pub struct SctMut<'a> {
    pub s: SignMatMut<'a>,
    pub c: &'a mut [f32],
    pub t: SignMatMut<'a>,
}

impl Sct {
    pub fn views(&self) -> impl IntoIterator<Item = (String, impl View + '_)> {
        let Self { s, c, t, .. } = self;
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

        impl<'a, T> Slice<'a, T> {
            pub fn new(slice: &'a [T]) -> Self {
                Self(slice, slice.len())
            }
        }

        impl<'a> DynView<'a> {
            pub fn new(value: impl View + 'a) -> Self {
                Self(Box::new(value))
            }
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
            ("s".to_owned(), DynView::new(Slice::new(&s))),
            ("t".to_owned(), DynView::new(Slice::new(&t))),
            ("c".to_owned(), DynView::new(Slice::new(c.as_slice()))),
        ]
    }
}

#![allow(non_snake_case)]

use core::marker::PhantomData;
use equator::assert;
use faer::SimpleEntity;
use reborrow::*;

#[doc(hidden)]
pub mod bitmagic;

pub mod inplace_sct;
pub mod sct;

trait Storage: Sized {}
impl Storage for u8 {}
impl Storage for u16 {}
impl Storage for u32 {}
impl Storage for u64 {}

pub struct MatRef<'a, T> {
    data: *const T,
    nrows: usize,
    ncols: usize,
    col_stride: isize,
    __marker: PhantomData<&'a T>,
}

pub struct MatMut<'a, T> {
    data: *mut T,
    nrows: usize,
    ncols: usize,
    col_stride: isize,
    __marker: PhantomData<&'a mut T>,
}

pub struct SignMatRef<'a> {
    storage: MatRef<'a, u64>,
    nrows: usize,
}

pub struct SignMatMut<'a> {
    storage: MatMut<'a, u64>,
    nrows: usize,
}

impl<'a, T> MatRef<'a, T> {
    #[inline]
    pub fn from_faer(mat: faer::MatRef<'a, T>) -> Self
    where
        T: SimpleEntity,
    {
        assert!(mat.row_stride() == 1);
        Self {
            data: mat.as_ptr(),
            nrows: mat.nrows(),
            ncols: mat.ncols(),
            col_stride: mat.col_stride(),
            __marker: PhantomData,
        }
    }

    #[inline]
    pub unsafe fn from_raw_parts(
        data: *const T,
        nrows: usize,
        ncols: usize,
        col_stride: isize,
    ) -> Self {
        Self {
            data,
            nrows,
            ncols,
            col_stride,
            __marker: PhantomData,
        }
    }

    #[track_caller]
    #[inline]
    pub fn from_col_major_slice(
        data: &'a [T],
        nrows: usize,
        ncols: usize,
        col_stride: usize,
    ) -> Self {
        if ncols == 0 {
            return Self {
                data: data.as_ptr(),
                nrows,
                ncols,
                col_stride: col_stride as isize,
                __marker: PhantomData,
            };
        }

        assert!(
            col_stride
                .checked_mul(ncols - 1)
                .and_then(|begin| begin.checked_add(nrows))
                .unwrap()
                <= data.len()
        );

        Self {
            data: data.as_ptr(),
            nrows,
            ncols,
            col_stride: col_stride as isize,
            __marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_ptr(self) -> *const T {
        self.data
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn col_stride(&self) -> isize {
        self.col_stride
    }

    #[inline]
    #[track_caller]
    pub fn submatrix(
        self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'a, T> {
        assert!(all(row_start <= self.nrows(), col_start <= self.ncols()));
        assert!(all(
            nrows <= self.nrows() - row_start,
            ncols <= self.ncols() - col_start,
        ));

        let col_stride = self.col_stride();
        let ptr = self.as_ptr();
        unsafe {
            Self::from_raw_parts(
                ptr.wrapping_add(row_start)
                    .wrapping_offset(col_start as isize * col_stride),
                nrows,
                ncols,
                col_stride,
            )
        }
    }

    #[inline]
    #[track_caller]
    pub fn col_as_slice(self, col: usize) -> &'a [T] {
        assert!(col < self.ncols());
        let col_stride = self.col_stride();
        let nrows = self.nrows();
        let ptr = self.as_ptr();
        unsafe { core::slice::from_raw_parts(ptr.offset(col as isize * col_stride), nrows) }
    }
}

impl<'a, T> MatMut<'a, T> {
    #[inline]
    pub fn from_faer(mat: faer::MatMut<'a, T>) -> Self
    where
        T: SimpleEntity,
    {
        let mut mat = mat;
        assert!(mat.row_stride() == 1);
        Self {
            data: mat.rb_mut().as_ptr_mut(),
            nrows: mat.nrows(),
            ncols: mat.ncols(),
            col_stride: mat.col_stride(),
            __marker: PhantomData,
        }
    }

    #[inline]
    pub unsafe fn from_raw_parts(
        data: *mut T,
        nrows: usize,
        ncols: usize,
        col_stride: isize,
    ) -> Self {
        Self {
            data,
            nrows,
            ncols,
            col_stride,
            __marker: PhantomData,
        }
    }

    #[track_caller]
    #[inline]
    pub fn from_col_major_slice(
        data: &'a mut [T],
        nrows: usize,
        ncols: usize,
        col_stride: usize,
    ) -> Self {
        if ncols == 0 {
            return Self {
                data: data.as_mut_ptr(),
                nrows,
                ncols,
                col_stride: col_stride as isize,
                __marker: PhantomData,
            };
        }

        assert!(all(
            col_stride >= nrows,
            col_stride
                .checked_mul(ncols - 1)
                .and_then(|begin| begin.checked_add(nrows))
                .unwrap()
                <= data.len(),
        ));

        Self {
            data: data.as_mut_ptr(),
            nrows,
            ncols,
            col_stride: col_stride as isize,
            __marker: PhantomData,
        }
    }

    #[inline]
    pub fn as_ptr(self) -> *const T {
        self.data
    }

    #[inline]
    pub fn as_ptr_mut(self) -> *mut T {
        self.data
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn col_stride(&self) -> isize {
        self.col_stride
    }

    #[inline]
    pub fn submatrix(
        self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'a, T> {
        self.into_const()
            .submatrix(row_start, col_start, nrows, ncols)
    }

    #[inline]
    #[track_caller]
    pub fn submatrix_mut(
        self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatMut<'a, T> {
        assert!(all(row_start <= self.nrows(), col_start <= self.ncols()));
        assert!(all(
            nrows <= self.nrows() - row_start,
            ncols <= self.ncols() - col_start,
        ));

        let col_stride = self.col_stride();
        let ptr = self.as_ptr_mut();
        unsafe {
            Self::from_raw_parts(
                ptr.wrapping_add(row_start)
                    .wrapping_offset(col_start as isize * col_stride),
                nrows,
                ncols,
                col_stride,
            )
        }
    }

    #[inline]
    pub fn col_as_slice(self, col: usize) -> &'a [T] {
        self.into_const().col_as_slice(col)
    }

    #[inline]
    #[track_caller]
    pub fn col_as_slice_mut(self, col: usize) -> &'a mut [T] {
        assert!(col < self.ncols());
        let col_stride = self.col_stride();
        let nrows = self.nrows();
        let ptr = self.as_ptr_mut();
        unsafe { core::slice::from_raw_parts_mut(ptr.offset(col as isize * col_stride), nrows) }
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col_mut(self, col: usize) -> (MatMut<'a, T>, MatMut<'a, T>) {
        assert!(col <= self.ncols());
        let nrows = self.nrows();
        let ncols = self.ncols();
        let col_stride = self.col_stride();
        let ptr = self.as_ptr_mut();

        unsafe {
            (
                Self::from_raw_parts(ptr, nrows, col, col_stride),
                Self::from_raw_parts(
                    ptr.wrapping_offset(col as isize * col_stride),
                    nrows,
                    ncols - col,
                    col_stride,
                ),
            )
        }
    }
}

impl<'a> SignMatRef<'a> {
    #[inline]
    #[track_caller]
    pub fn from_storage(storage: MatRef<'a, u64>, nrows: usize) -> Self {
        assert!(storage.nrows() * 64 >= nrows);
        Self { storage, nrows }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.storage.ncols()
    }

    #[inline]
    pub fn storage(self) -> MatRef<'a, u64> {
        self.storage
    }

    #[inline]
    fn storage_as<T: Storage>(self) -> MatRef<'a, T> {
        unsafe {
            MatRef::from_raw_parts(
                self.storage.as_ptr() as *const T,
                self.nrows().div_ceil(core::mem::size_of::<T>()),
                self.ncols(),
                self.storage.col_stride()
                    * (core::mem::size_of::<u64>() / core::mem::size_of::<T>()) as isize,
            )
        }
    }
}

impl<'a> SignMatMut<'a> {
    #[inline]
    #[track_caller]
    pub fn from_storage(storage: MatMut<'a, u64>, nrows: usize) -> Self {
        assert!(storage.nrows() * 64 >= nrows);
        Self { storage, nrows }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.storage.ncols()
    }

    #[inline]
    pub fn storage(self) -> MatRef<'a, u64> {
        self.storage.into_const()
    }

    #[inline]
    pub fn storage_mut(self) -> MatMut<'a, u64> {
        self.storage
    }

    #[inline]
    #[track_caller]
    pub fn split_at_col_mut(self, col: usize) -> (SignMatMut<'a>, SignMatMut<'a>) {
        let nrows = self.nrows();
        let (left, right) = self.storage_mut().split_at_col_mut(col);
        (
            Self::from_storage(left, nrows),
            Self::from_storage(right, nrows),
        )
    }
}

impl<T> Clone for MatRef<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for MatRef<'_, T> {}
impl<'short, T> Reborrow<'short> for MatRef<'_, T> {
    type Target = MatRef<'short, T>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short, T> ReborrowMut<'short> for MatRef<'_, T> {
    type Target = MatRef<'short, T>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'a, T> IntoConst for MatRef<'a, T> {
    type Target = MatRef<'a, T>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'short, T> Reborrow<'short> for MatMut<'_, T> {
    type Target = MatRef<'short, T>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        MatRef {
            data: self.data,
            nrows: self.nrows,
            ncols: self.ncols,
            col_stride: self.col_stride,
            __marker: PhantomData,
        }
    }
}
impl<'short, T> ReborrowMut<'short> for MatMut<'_, T> {
    type Target = MatMut<'short, T>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        MatMut {
            data: self.data,
            nrows: self.nrows,
            ncols: self.ncols,
            col_stride: self.col_stride,
            __marker: PhantomData,
        }
    }
}
impl<'a, T> IntoConst for MatMut<'a, T> {
    type Target = MatRef<'a, T>;

    #[inline]
    fn into_const(self) -> Self::Target {
        MatRef {
            data: self.data,
            nrows: self.nrows,
            ncols: self.ncols,
            col_stride: self.col_stride,
            __marker: PhantomData,
        }
    }
}

impl Copy for SignMatRef<'_> {}
impl Clone for SignMatRef<'_> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'short> Reborrow<'short> for SignMatRef<'_> {
    type Target = SignMatRef<'short>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        *self
    }
}
impl<'short> ReborrowMut<'short> for SignMatRef<'_> {
    type Target = SignMatRef<'short>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        *self
    }
}
impl<'a> IntoConst for SignMatRef<'a> {
    type Target = SignMatRef<'a>;

    #[inline]
    fn into_const(self) -> Self::Target {
        self
    }
}

impl<'short> Reborrow<'short> for SignMatMut<'_> {
    type Target = SignMatRef<'short>;

    #[inline]
    fn rb(&'short self) -> Self::Target {
        SignMatRef {
            storage: self.storage.rb(),
            nrows: self.nrows,
        }
    }
}
impl<'short> ReborrowMut<'short> for SignMatMut<'_> {
    type Target = SignMatMut<'short>;

    #[inline]
    fn rb_mut(&'short mut self) -> Self::Target {
        SignMatMut {
            storage: self.storage.rb_mut(),
            nrows: self.nrows,
        }
    }
}
impl<'a> IntoConst for SignMatMut<'a> {
    type Target = SignMatRef<'a>;

    #[inline]
    fn into_const(self) -> Self::Target {
        SignMatRef {
            storage: self.storage.into_const(),
            nrows: self.nrows,
        }
    }
}

impl<T> core::ops::Index<(usize, usize)> for MatRef<'_, T> {
    type Output = T;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(all(row < self.nrows(), col < self.ncols()));
        unsafe {
            &*(self
                .as_ptr()
                .add(row)
                .offset(col as isize * self.col_stride()))
        }
    }
}

impl<T> core::ops::Index<(usize, usize)> for MatMut<'_, T> {
    type Output = T;

    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(all(row < self.nrows(), col < self.ncols()));
        unsafe {
            &*(self
                .rb()
                .as_ptr()
                .add(row)
                .offset(col as isize * self.col_stride()))
        }
    }
}

impl<T> core::ops::IndexMut<(usize, usize)> for MatMut<'_, T> {
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(all(row < self.nrows(), col < self.ncols()));
        unsafe {
            &mut *(self
                .rb_mut()
                .as_ptr_mut()
                .add(row)
                .offset(col as isize * self.col_stride()))
        }
    }
}

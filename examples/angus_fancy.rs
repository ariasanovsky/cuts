use core::f32;

use clap::Parser;
use cuts::{inplace_sct::CutHelper, SignMatMut};
use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use equator::assert;
use faer::{linalg::temp_mat_req, solvers::SolverCore, Col, ColMut, ColRef, Mat, MatMut, MatRef};
use image::{open, ImageBuffer, Rgb};
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};
use reborrow::{Reborrow, ReborrowMut};

#[derive(Debug, Parser)]
#[command(name = "Angus")]
#[command(about = "Approximates an image with signed cuts", long_about = None)]
struct Args {
    /// Input directory containing `safetensors`
    #[arg(short = 'i')]
    input: std::path::PathBuf,
    /// Output directory for new `safetensors`
    #[arg(short = 'o')]
    output: std::path::PathBuf,
    /// The width
    #[arg(short = 'w')]
    width: usize,
    /// Render an image at step `{1, 1+s, 1+2s, ...}`
    #[arg(short = 's')]
    step: usize,
    // /// The number of tensors to process in parallel
    // #[arg(short = 't')]
    // threads: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let Args {
        input,
        output,
        width,
        step,
        // threads: _,
    } = Args::try_parse()?;
    let stem = input
        .file_stem()
        .unwrap()
        .to_os_string()
        .into_string()
        .unwrap();
    let img = open(input)?.into_rgb8();
    let (nrows, ncols) = img.dimensions();
    let (nrows, ncols): (usize, usize) = (nrows as _, ncols as _);
    let width_bits = 32 + 3 + nrows + ncols;
    let nbytes = (width * width_bits).div_ceil(8);
    dbg!(width, nbytes);
    let bytes = (0..3)
        .flat_map(|c| img.pixels().map(move |p| p.0[c]))
        .collect::<Vec<_>>();
    let bytes = RgbTensor::new(bytes, nrows, ncols);
    let a = bytes.clone().convert(|c| c as f32);

    let rng = &mut StdRng::seed_from_u64(0);
    let mut smat: SignMatrix = SignMatrix::new(nrows);
    let mut tmat: SignMatrix = SignMatrix::new(ncols);
    let mut kmat: SignMatrix = SignMatrix::new(3);
    let init_norm = a.col().norm_l2();
    let mut r = a.clone();
    let mut mem = GlobalPodBuffer::new(
        StackReq::new::<u64>(Ord::max(nrows, ncols)).and(temp_mat_req::<f32>(1, 1).unwrap()),
    );
    let mut stack = PodStack::new(&mut mem);

    for w in 0..width {
        let r_gammas: [Mat<f32>; 4] = GAMMA.map(|k| r.combine_colors(&k));
        let cuts: [(Col<f32>, Col<f32>); 4] =
            core::array::from_fn(|i| greedy_cut(r_gammas[i].as_ref(), rng, stack.rb_mut()));
        let mut coefficients: [(f32, Col<f32>); 4] = core::array::from_fn(|i| {
            let mut smat = smat.clone();
            smat.push(cuts[i].0.as_slice());
            let mut tmat = tmat.clone();
            tmat.push(cuts[i].1.as_slice());
            let mut kmat = kmat.clone();
            kmat.push(&GAMMA[i]);
            regress(&a, &smat, &tmat, &kmat)
        });
        let i_max = coefficients
            .iter()
            .position_max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap();
        smat.push(cuts[i_max].0.as_slice());
        tmat.push(cuts[i_max].1.as_slice());
        kmat.push(GAMMA[i_max].as_slice());
        let c = coefficients[i_max].1.as_mut();
        r = a.minus(&smat, &tmat, &kmat, c.rb());
        let improvements = improve_signs_then_coefficients_repeatedly(
            &a,
            &mut r,
            &mut smat,
            &mut tmat,
            &mut kmat,
            c,
            stack.rb_mut(),
        );
        let rel_error = r.col().norm_l2() / init_norm;
        let w = w + 1;
        println!(
            "({}, {}, {}, {}),",
            w, improvements.0, improvements.1, rel_error
        );
        if w % step == 1 || w == width {
            let outpath = output.join(format!("{stem}-{w:04b}.jpg"));
            let approx = a.col() - r.col();
            let output = ImageBuffer::from_fn(nrows as _, ncols as _, |i, j| {
                let i = i as usize;
                let j = j as usize;
                let ij = i + nrows * j;
                let rgb: [u8; 3] = core::array::from_fn(|c| to_u8(approx[c * nrows * ncols + ij]));
                Rgb(rgb)
            });
            output.save(outpath)?;
        }
    }
    Ok(())
}

fn u8_error(c: u8, x: f32) -> f32 {
    match c {
        0 => f32::max(0.0, x - 0.5),
        255 => f32::max(0.0, 254.5 - x),
        c => f32::max(0.0, f32::abs(x - c as f32) - 0.5),
    }
}

fn u8_errors<'a>(bytes: &'a [u8], mat: MatRef<'a, f32>) -> impl Iterator<Item = f32> + 'a {
    let (nrows, ncols) = mat.shape();
    (0..ncols).flat_map(move |j| {
        (0..nrows).map(move |i| {
            let ij = i + nrows * j;
            let c = bytes[ij];
            let x = mat[(i, j)];
            u8_error(c, x)
        })
    })
}

fn to_u8(x: f32) -> u8 {
    assert!(x.is_finite());
    let x = x.clamp(0.0, 255.0);
    x.round() as _
}

fn greedy_cut(mat: MatRef<f32>, rng: &mut impl rand::Rng, stack: PodStack) -> (Col<f32>, Col<f32>) {
    let (nrows, ncols) = mat.shape();
    let mut s = Col::from_fn(nrows, |_| if rng.gen() { -1.0f32 } else { 1.0 });
    let mut t = Col::from_fn(ncols, |_| if rng.gen() { -1.0f32 } else { 1.0 });
    let two_remainder = faer::scale(2.0f32) * mat.rb();
    let two_remainder_transposed = two_remainder.transpose().to_owned();
    // let mut s_ones = vec![0u64; nrows.div_ceil(64)].into_boxed_slice();
    // let mut t_ones = vec![0u64; ncols.div_ceil(64)].into_boxed_slice();
    let (bit_rows, bit_cols) = (nrows.div_ceil(64), ncols.div_ceil(64));
    let mut s_ones = vec![0u64; bit_rows].into_boxed_slice();
    let mut t_ones = vec![0u64; bit_cols].into_boxed_slice();
    let s_ones = cuts::MatMut::from_col_major_slice(&mut s_ones, bit_rows, 1, bit_rows);
    let s_ones = SignMatMut::from_storage(s_ones, nrows);
    let t_ones = cuts::MatMut::from_col_major_slice(&mut t_ones, bit_cols, 1, bit_cols);
    let t_ones = SignMatMut::from_storage(t_ones, ncols);

    let _ = improve_greedy_cut(
        two_remainder.as_ref(),
        two_remainder_transposed.as_ref(),
        s.as_mut(),
        t.as_mut(),
        s_ones,
        t_ones,
        stack,
    );
    (s, t)
}

fn improve_greedy_cut(
    two_remainder: MatRef<f32>,
    two_remainder_transposed: MatRef<f32>,
    s: ColMut<f32>,
    t: ColMut<f32>,
    mut s_ones: SignMatMut,
    mut t_ones: SignMatMut,
    stack: PodStack,
) -> (f32, usize) {
    let (nrows, ncols) = two_remainder.shape();
    {
        let s_ones = s_ones.rb_mut().storage_mut().col_as_slice_mut(0);
        s.rb().iter().enumerate().for_each(|(i, si)| {
            let pos = i / 64;
            let rem = i % 64;
            let signs = &mut s_ones[pos];
            if si.is_sign_negative() {
                *signs |= 1 << rem
            }
        });
        let t_ones = t_ones.rb_mut().storage_mut().col_as_slice_mut(0);
        t.rb().iter().enumerate().for_each(|(i, ti)| {
            let pos = i / 64;
            let rem = i % 64;
            let signs = &mut t_ones[pos];
            if ti.is_sign_negative() {
                *signs |= 1 << rem
            }
        });
    }
    let mut helper: CutHelper = CutHelper::new_with_st(
        two_remainder.as_ref(),
        two_remainder_transposed.as_ref(),
        s_ones
            .rb()
            .storage()
            .col_as_slice(0)
            .iter()
            .copied()
            .collect(),
        t_ones
            .rb()
            .storage()
            .col_as_slice(0)
            .iter()
            .copied()
            .collect(),
    );
    let (u64_rows, u64_cols) = (nrows.div_ceil(64), ncols.div_ceil(64));
    // let s_mat = cuts::MatMut::from_col_major_slice(s_ones, u64_rows, 1, u64_rows);
    // let mut s_mat = SignMatMut::from_storage(s_mat, nrows);
    let mut c = [0.0f32];
    let mut c = faer::col::from_slice_mut(&mut c);
    // let t_mat = cuts::MatMut::from_col_major_slice(t_ones, u64_cols, 1, u64_cols);
    // let mut t_mat = SignMatMut::from_storage(t_mat, ncols);
    let new_c = helper.cut_mat_inplace(
        two_remainder.as_ref(),
        two_remainder_transposed.as_ref(),
        s_ones.rb_mut(),
        c.rb_mut(),
        t_ones.rb_mut(),
        core::usize::MAX,
        stack,
    );
    let mut improved_signs = 0;
    let s_signs = s_ones.storage().col_as_slice(0);
    s.iter_mut()
        .zip(s_signs.iter().flat_map(|&signs| {
            (0..64).map(move |i| if signs & (1 << i) != 0 { -1.0f32 } else { 1.0 })
        }))
        .for_each(|(si, s_sign)| {
            if *si != s_sign {
                improved_signs += 1;
                *si = s_sign
            }
        });
    let t_signs = t_ones.storage().col_as_slice(0);
    t.iter_mut()
        .zip(t_signs.iter().flat_map(|&signs| {
            (0..64).map(move |i| if signs & (1 << i) != 0 { -1.0f32 } else { 1.0 })
        }))
        .for_each(|(ti, t_sign)| {
            if *ti != t_sign {
                improved_signs += 1;
                *ti = t_sign
            }
        });
    (new_c, improved_signs)
}

fn regress(
    a: &RgbTensor<f32>,
    smat: &SignMatrix,
    tmat: &SignMatrix,
    kmat: &SignMatrix,
) -> (f32, Col<f32>) {
    let smat = smat.as_mat_ref();
    let tmat = tmat.as_mat_ref();
    let kmat = kmat.as_mat_ref();
    let sts = smat.transpose() * smat;
    let ttt = tmat.transpose() * tmat;
    let ktk = kmat.transpose() * kmat;
    let width = sts.nrows();
    let xtx = Mat::from_fn(width, width, |i, j| sts[(i, j)] * ttt[(i, j)] * ktk[(i, j)]);
    // println!("sts = {sts:?}");
    // println!("ttt = {ttt:?}");
    // println!("ktk = {ktk:?}");
    // println!("xtx = {xtx:?}");
    let inv = xtx.cholesky(faer::Side::Upper).unwrap().inverse();
    let xta: Col<f32> = Col::from_fn(width, |i| {
        let dots: [f32; 3] =
            core::array::from_fn(|c| smat.col(i).transpose() * a.mat(c) * tmat.col(i));
        kmat.col(i).transpose() * faer::col::from_slice(dots.as_slice())
    });
    let coefficients = inv * &xta;
    let aw_norm = coefficients.transpose() * xta;
    (aw_norm, coefficients)
}

fn improve_signs_then_coefficients_repeatedly(
    a: &RgbTensor<f32>,
    r: &mut RgbTensor<f32>,
    smat: &mut SignMatrix,
    tmat: &mut SignMatrix,
    kmat: &mut SignMatrix,
    mut c: ColMut<f32>,
    mut stack: PodStack,
) -> (usize, usize) {
    let (nrows, ncols) = a.shape();
    let width = c.nrows();
    let mut total_iterations = 0;
    let mut coefficient_updates = 0;
    loop {
        let mut improved = false;
        for j in 0..width {
            let mut c_j = c.rb().to_owned();
            c_j[j] = 0.0;
            let r_j = a.minus(smat, tmat, kmat, c_j.as_ref());
            let k_j = kmat.as_mat_ref().col(j);
            let k = &[k_j[0], k_j[1], k_j[2]];
            let r_j = r_j.combine_colors(k);
            let mut s_j = smat.as_mat_mut().col_mut(j);
            let mut t_j = tmat.as_mat_mut().col_mut(j);
            let two_remainder = faer::scale(2.0f32) * r_j.as_ref();
            let two_remainder_transposed = two_remainder.transpose().to_owned();
            let (bit_rows, bit_cols) = (nrows.div_ceil(64), ncols.div_ceil(64));
            let mut s_ones = vec![0u64; bit_rows].into_boxed_slice();
            let mut t_ones = vec![0u64; bit_cols].into_boxed_slice();
            let s_ones = cuts::MatMut::from_col_major_slice(&mut s_ones, bit_rows, 1, bit_rows);
            let s_ones = SignMatMut::from_storage(s_ones, nrows);
            let t_ones = cuts::MatMut::from_col_major_slice(&mut t_ones, bit_cols, 1, bit_cols);
            let t_ones = SignMatMut::from_storage(t_ones, ncols);
            let (_, iterations) = improve_greedy_cut(
                two_remainder.as_ref(),
                two_remainder_transposed.as_ref(),
                s_j.as_mut(),
                t_j.as_mut(),
                s_ones,
                t_ones,
                stack.rb_mut(),
            );
            if iterations != 0 {
                total_iterations += iterations;
                let (_, c_new) = regress(a, smat, tmat, kmat);
                *r = a.minus(smat, tmat, kmat, c_new.as_ref());
                improved = true;
            }
        }
        if improved {
            let (_, c_new) = regress(a, &smat, &tmat, kmat);
            c.rb_mut()
                .iter_mut()
                .zip(c_new.iter())
                .for_each(|(c, c_new)| *c = *c_new);
            *r = a.minus(smat, tmat, kmat, c.rb());
            coefficient_updates += 1;
        } else {
            return (total_iterations, coefficient_updates);
        }
    }
}

// layout: [a0, a1, a2] where each ai is column-major
#[derive(Clone)]
struct RgbTensor<T> {
    nrows: usize,
    ncols: usize,
    data: Vec<T>,
}

impl<T> RgbTensor<T> {
    fn new(data: Vec<T>, nrows: usize, ncols: usize) -> Self {
        assert!(data.len() == nrows * ncols * 3);
        Self { nrows, ncols, data }
    }

    fn convert<U>(self, f: impl Fn(T) -> U) -> RgbTensor<U> {
        let Self { nrows, ncols, data } = self;
        RgbTensor {
            nrows,
            ncols,
            data: data.into_iter().map(f).collect(),
        }
    }

    fn color(&self, c: usize) -> &[T] {
        assert!(c < 3);
        let num = self.nrows * self.ncols;
        &self.data[num * c..][..num]
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

impl RgbTensor<f32> {
    fn mat(&self, c: usize) -> MatRef<f32> {
        faer::mat::from_column_major_slice(self.color(c), self.nrows, self.ncols)
    }

    fn col(&self) -> ColRef<f32> {
        faer::col::from_slice(&self.data)
    }

    fn combine_colors(&self, &[k0, k1, k2]: &[f32; 3]) -> Mat<f32> {
        faer::scale(k0) * self.mat(0)
            + faer::scale(k1) * self.mat(1)
            + faer::scale(k2) * self.mat(2)
    }

    fn minus(
        &self,
        smat: &SignMatrix,
        tmat: &SignMatrix,
        kmat: &SignMatrix,
        c: ColRef<f32>,
    ) -> Self {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let mut r = self.clone();
        let mut r_col = faer::col::from_slice_mut(r.data.as_mut_slice());
        for (((s, t), k), coe) in smat
            .as_mat_ref()
            .col_iter()
            .zip(tmat.as_mat_ref().col_iter())
            .zip(kmat.as_mat_ref().col_iter())
            .zip(c.iter())
        {
            for c in 0..3 {
                for j in 0..ncols {
                    for i in 0..nrows {
                        let cji = i + nrows * j + c * nrows * ncols;
                        r_col[cji] -= coe * k[c] * t[j] * s[i]
                    }
                }
            }
        }
        r
    }
}

// impl<T> Index<(usize, usize, usize)> for RgbTensor<T> {
//     type Output = T;

//     fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
//         todo!()
//     }
// }

#[derive(Clone)]
struct SignMatrix {
    nrows: usize,
    ncols: usize,
    data: Vec<f32>,
}

impl SignMatrix {
    fn new(nrows: usize) -> Self {
        Self {
            data: vec![],
            nrows,
            ncols: 0,
        }
    }

    fn push(&mut self, col: &[f32]) {
        assert!(col.len() == self.nrows);
        self.data.extend_from_slice(col);
        self.ncols += 1
    }

    fn as_mat_ref(&self) -> MatRef<f32> {
        faer::mat::from_column_major_slice(self.data.as_slice(), self.nrows, self.ncols)
    }

    fn as_mat_mut(&mut self) -> MatMut<f32> {
        faer::mat::from_column_major_slice_mut(self.data.as_mut_slice(), self.nrows, self.ncols)
    }
}

const GAMMA: [[f32; 3]; 4] = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0],
];

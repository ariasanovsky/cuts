use std::ops::Index;

use equator::assert;
use aligned_vec::ABox;
use clap::Parser;
use cuts::{inplace_sct::CutHelper, sct::{Sct, SctMut, SctRef}, sct_tensor::{Cut, Remainder}};
use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use faer::{linalg::temp_mat_req, mat::AsMatRef, solvers::SolverCore, Col, ColRef, Mat, MatRef};
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
    #[arg(short = 'c')]
    compression_rate: f64,
    /// The number of tensors to process in parallel
    #[arg(short = 't')]
    threads: Option<usize>,
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
        Self { nrows, ncols, data: data }
    }

    fn convert<U>(self, f: impl Fn(T) -> U) -> RgbTensor<U> {
        let Self { nrows, ncols, data } = self;
        RgbTensor { nrows, ncols, data: data.into_iter().map(f).collect() }
    }

    fn color(&self, c: usize) -> &[T] {
        assert!(c < 3);
        let num = self.nrows * self.ncols;
        &self.data[num * c..][..num]
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
        Self { data: vec![], nrows, ncols: 0 }
    }

    fn push(&mut self, col: &[f32]) {
        assert!(col.len() == self.nrows);
        self.data.extend_from_slice(col);
        self.ncols += 1
    }

    fn mat(&self) -> MatRef<f32> {
        faer::mat::from_column_major_slice(self.data.as_slice(), self.nrows, self.ncols)
    }
}

fn main() -> eyre::Result<()> {
    let Args {
        input,
        output,
        compression_rate,
        threads: _,
    } = Args::try_parse()?;
    let img = open(input)?.into_rgb8();
    let (nrows, ncols) = img.dimensions();
    let (nrows, ncols): (usize, usize) = (nrows as _, ncols as _);
    let original_bits = nrows * ncols * 24;
    let width_bits = 32 + 3 + nrows + ncols;
    let width = (original_bits as f64 * compression_rate / width_bits as f64) as usize;
    let nbytes = (width * width_bits).div_ceil(8);
    dbg!(width, nbytes);
    let bytes = (0..3)
        .flat_map(|c| img
            .pixels()
            .map(move |p| p.0[c])
        ).collect::<Vec<_>>();
    let bytes = RgbTensor::new(bytes, nrows, ncols);
    let a = bytes.clone().convert(|c| c as f32);

    let rng = &mut StdRng::seed_from_u64(0);
    let mut smat: SignMatrix = SignMatrix::new(nrows);
    let mut tmat: SignMatrix = SignMatrix::new(ncols);
    let mut kvecs: SignMatrix = SignMatrix::new(3);
    let init_norm = a.col().squared_norm_l2();
    let r = a.clone();
    const GAMMA: [[f32; 3]; 4] = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
    ];
    for w in 0..width {
        let r_gammas: [Mat<f32>; 4] = GAMMA.map(|k| {
            r.combine_colors(&k)
        });
        let cuts: [(Col<f32>, Col<f32>); 4] = core::array::from_fn(|i| {
            greedy_cut(&r_gammas[i], rng)
        });
        let coefficients: [(f32, Col<f32>); 4] = core::array::from_fn(|i| {
            regress(&a, &smat, &tmat, &cuts[i])
        });
        todo!()
    //     let coefficients: [(f32, Col<f32>, Col<f32>, Col<f32>); 4] = core::array::from_fn(|i| {
    //         let g = &GAMMA[i];
    //         let two_remainder =
    //             faer::scale(2.0 * g[0]) * &remainders[0]
    //             + faer::scale(2.0 * g[1]) * &remainders[1]
    //             + faer::scale(2.0 * g[2]) * &remainders[2]
    //         ;
    //         let two_remainder_transposed = two_remainder.transpose().to_owned();
    //         let mut helper = CutHelper::new(two_remainder.as_ref(), two_remainder_transposed.as_ref());
    //         let mut sct_block = Sct::new(nrows, ncols, 1);
    //         let SctMut { mut s, c, mut t } = sct_block.as_mut();
    //         let rng = &mut rngs[i];
    //         let mut mem = GlobalPodBuffer::new(
    //             StackReq::new::<u64>(Ord::max(nrows, ncols))
    //                 .and(temp_mat_req::<f32>(1, 1).unwrap()),
    //         );
    //         let mut stack = PodStack::new(&mut mem);
    //         let _ = helper.cut_mat(
    //             two_remainder.as_ref(),
    //             two_remainder_transposed.as_ref(),
    //             s.rb_mut().split_at_col_mut(1).0,
    //             faer::col::from_slice_mut(&mut c[..1]),
    //             t.rb_mut().split_at_col_mut(1).0,
    //             rng,
    //             usize::MAX,
    //             stack.rb_mut(),
    //         );
    //         let SctRef { s, c: _, t } = sct_block.as_ref();
    //         let mut svecs = svecs.clone();
    //         let mut tvecs = tvecs.clone();
    //         let mut kvecs = kvecs.clone();
    //         let to_col = |bytes: &[u64], nrows: usize| -> Col<f32> {
    //             assert!(bytes.len() * 64 >= nrows);
    //             let mut signs = Col::zeros(nrows);
    //             for (i, sign) in bytes.iter().flat_map(|&bits| {
    //                 (0..64).map(move |i| if (1 << i) & bits != 0 {
    //                     -1.0
    //                 } else {
    //                     1.0
    //                 })
    //             }).take(nrows).enumerate() {
    //                 signs[i] = sign
    //             }
    //             signs
    //         };
    //         svecs.push(to_col(s.storage().col_as_slice(0), nrows));
    //         tvecs.push(to_col(t.storage().col_as_slice(0), ncols));
    //         kvecs.push(faer::col::from_slice(g).to_owned());
    //         let mut xtx: Mat<f32> = Mat::zeros(w + 1, w + 1);
    //         for i in 0..=w {
    //             for j in 0..=w {
    //                 let sij = svecs[i].transpose() * &svecs[j] / (nrows as f32);
    //                 let tij = tvecs[i].transpose() * &tvecs[j] / (ncols as f32);
    //                 let kij = kvecs[i].transpose() * &kvecs[j] / 3.0;
    //                 xtx[(i, j)] = sij * tij * kij;
    //             }
    //         }
    //         // println!("{xtx:?}");
    //         let xta: Col<f32> = Col::from_fn(w + 1, |i| {
    //             (0..3).map(|u| {
    //                 let at = &mats[u] * &tvecs[i];
    //                 let sat = (svecs[i].transpose() * at) / (nrows * ncols) as f32;
    //                 (g[u] * sat) / 3.0
    //             }).sum::<f32>()
    //         });
    //         // println!("xta = {xta:?}");
    //         let inv = xtx.cholesky(faer::Side::Lower).unwrap().inverse();
    //         let c = inv * &xta;
    //         // println!("c = {c:?}");
    //         let a_norm = c.transpose() * xta;
    //         println!("a_norm = {a_norm}");
    //         (a_norm, c, svecs.pop().unwrap(), tvecs.pop().unwrap())
    //     });
    //     let i_max = coefficients.iter().position_max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
    //     assert!(i_max == 0);
    //     let g = &GAMMA[i_max];
    //     let (_, c, s, t) = &coefficients[i_max];
    //     svecs.push(s.to_owned());
    //     tvecs.push(t.to_owned());
    //     kvecs.push(faer::col::from_slice(g).to_owned());
    //     st_mats.push(s * t.transpose());
    //     for u in 0..3 {
    //         let mut r = mats[u].clone();
    //         for ((c, st), g) in c.as_slice().iter().zip(st_mats.iter()).zip(kvecs.iter()) {
    //             r -= faer::scale(g[u] * c) * st;
    //         }
    //         remainders[u] = r;
    //     }
    //     println!("{w}: srt = {:?}", remainders.iter().map(|r| r.squared_norm_l2()).sum::<f32>().sqrt() / init_norm);
    //     // for u in 0..3 {
    //     //     let r = &remainders[u];
    //     //     println!("||r[{u}]|| = {}", r.squared_norm_l2());
    //     //     let rt = r * tvecs.last().unwrap();
    //     //     let srt = svecs.last().unwrap().transpose() * rt;
    //     //     println!("srt[{u}] = {srt}")
    //     // }
    //     println!("{w}, {g:?}")
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
    (0..ncols).flat_map(move |j| (0..nrows).map(move |i| {
        let ij = i + nrows * j;
        let c = bytes[ij];
        let x = mat[(i, j)];
        u8_error(c, x)
    }))
}

fn greedy_cut(mat: &Mat<f32>, rng: &mut impl rand::Rng) -> (Col<f32>, Col<f32>) {
    todo!()
}

fn regress(
    a: &RgbTensor<f32>,
    smat: &SignMatrix,
    tmat: &SignMatrix,
    cut: &(Col<f32>, Col<f32>)
) -> (f32, Col<f32>) {
    todo!()
}
use aligned_vec::ABox;
use clap::Parser;
use cuts::sct_tensor::{Cut, Remainder};
use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use faer::{mat::AsMatRef, Col, Mat, MatRef};
use image::{open, ImageBuffer, Rgb};
use rand::{rngs::StdRng, SeedableRng};
use reborrow::ReborrowMut;

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
    let mut mats: [Mat<f32>; 3] = core::array::from_fn(|_| Mat::zeros(nrows, ncols));
    let mut bytes: [_; 3] = core::array::from_fn(|_| aligned_vec::avec![0; nrows * ncols].into_boxed_slice());
    for j in 0..ncols {
        for i in 0..nrows {
            let Rgb(p) = img.get_pixel(i as _, j as _);
            p
                .iter()
                .zip(mats.iter_mut())
                .zip(bytes.iter_mut())
                .for_each(|((&c, mat), bytes)| {
                    mat[(i, j)] = c as f32;
                    let ij = i + nrows * j;
                    bytes[ij] = c;
                })
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
    (0..ncols).flat_map(move |j| (0..nrows).map(move |i| {
        let ij = i + nrows * j;
        let c = bytes[ij];
        let x = mat[(i, j)];
        u8_error(c, x)
    }))
}
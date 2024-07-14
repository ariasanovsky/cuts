use clap::Parser;
use cuts_v2::sct_tensor::{Cut, Remainder};
use dyn_stack::{GlobalPodBuffer, PodStack, StackReq};
use faer::Col;
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
    /// The number of blocks to use in delayed matmul
    #[arg(short = 'b')]
    block_size: usize,
    /// The number of tensors to process in parallel
    #[arg(short = 't')]
    threads: Option<usize>,
}

fn main() -> eyre::Result<()> {
    let Args {
        input,
        output,
        compression_rate,
        block_size: _,
        threads: _,
    } = Args::try_parse()?;
    let img = open(input)?.into_rgb8();
    let (nrows, ncols) = img.dimensions();
    let (nrows, ncols): (usize, usize) = (nrows as _, ncols as _);
    let dim = [3, nrows, ncols];
    let stride = [1, dim[0], dim[0] * dim[1]];
    let t = img.pixels().flat_map(|&Rgb(p)| {
        [p[0] as f32, p[1] as f32, p[2] as f32]
    }).collect::<Vec<_>>();
    let init_norm = faer::col::from_slice(&t).norm_l2();
    let total_dim = t.len();
    let mut remainder = Remainder::new(&t, &dim, &stride);
    let original_bits = total_dim * 8;
    let width_bits = remainder.width_bits();
    let width = original_bits as f64 * compression_rate / width_bits as f64;
    let width = width as usize;
    let nbytes = (width * width_bits).div_ceil(8);
    dbg!(width, nbytes);
    let mut mem = {
        let sign = StackReq::new::<u64>(total_dim.div_ceil(64));
        let sign_f32 = StackReq::new::<f32>(total_dim);
        GlobalPodBuffer::new(sign.and(sign_f32))
    };
    let mut cut = Cut::new(&dim, remainder.two_mats(), PodStack::new(&mut mem));
    let mut stack = PodStack::new(&mut mem);
    let rng = &mut StdRng::seed_from_u64(0);
    // let normalization = ((255 * 255 * nrows * ncols * 3) as f32).sqrt();
    let normalization = init_norm;
    let mut approx = Col::zeros(total_dim);
    for w in 0..width {
        remainder.fill_matrices();
        remainder.cut(&mut cut, rng, stack.rb_mut());
        {
            let mut blowup = faer::Col::<f32>::zeros(total_dim);
            cut.blowup_mul_add(blowup.as_slice_mut(), 1.0, stack.rb_mut());
            approx += faer::scale(cut.c() / total_dim as f32) * &blowup;
            let dot = faer::row::from_slice(&remainder.t()) * blowup;
            let _err = (dot - cut.c()).abs() / f32::max(cut.c().abs(), dot.abs());
        }
        remainder.update(&cut, stack.rb_mut());
        {
            let curr_error = faer::col::from_slice(&remainder.t()).norm_l2() / normalization;
            if w % 10 == 0 {
                println!("{w}: {curr_error}");
            }
        }
    }
    // let mut rgb: [Mat<f32>; 3] = core::array::from_fn(|_| Mat::zeros(nrows, ncols));
    // // let mut rmat: Mat<f32> = Mat::zeros(nrows, ncols);
    // // let mut gmat: Mat<f32> = Mat::zeros(nrows, ncols);
    // // let mut bmat: Mat<f32> = Mat::zeros(nrows, ncols);
    // for (row, col, &Rgb([r, g, b])) in img.enumerate_pixels() {
    //     rgb[0][(row as _, col as _)] = r as _;
    //     rgb[1][(row as _, col as _)] = g as _;
    //     rgb[2][(row as _, col as _)] = b as _;
    // }
    // let nbits_per_color = nrows * ncols * 8;
    // let nbits_per_cut = nrows + ncols + 32;
    // dbg!(nbits_per_color, nbits_per_cut);
    // let width = compression_rate * nbits_per_color as f64 / nbits_per_cut as f64;
    // let width = width as usize;
    // let mut sct = rgb.map(|mat| {
    //     SctHelper::new(mat.as_ref(), block_size, width)
    // });
    // let mats: [Mat<f32>; 3] = core::array::from_fn(|c| {
    //     let mut mat = &mut sct[c];
    //     let ref mut rng = StdRng::seed_from_u64(0);
    //     let normalization = (255 * 255 * nrows * ncols) as f32;
    //     for w in 0..width {
    //         let cut = mat.cut(rng);
    //         dbg!(w, (mat.squared_norm_l2() / normalization).sqrt());
    //     }
    //     let r = mat.expand();
    //     r
    // });
    let out = ImageBuffer::from_fn(nrows as _, ncols as _, |i, j| {
        let i = i as usize;
        let j = j as usize;
        Rgb(core::array::from_fn(|c| {
            let ijc = c * stride[0] + i * stride[1] + j * stride[2];
            to_u8(approx[ijc])
        }))
    });
    out.save(output)?;
    dbg!(nbytes);
    Ok(())
}

fn to_u8(x: f32) -> u8 {
    assert!(x.is_finite());
    let x = x.clamp(0.0, 255.0);
    x.round() as _
}

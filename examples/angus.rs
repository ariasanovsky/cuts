use clap::Parser;
use cuts::sct_tensor::{Cut, Remainder};
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
    #[arg(short = 'w')]
    width: usize,
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
        width,
        block_size,
        threads: _,
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
    let dim = [3, nrows, ncols];
    let stride = [1, dim[0], dim[0] * dim[1]];
    let t = img
        .pixels()
        .flat_map(|&Rgb(p)| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect::<Vec<_>>();
    let init_norm = faer::col::from_slice(&t).norm_l2();
    let total_dim = t.len();
    let mut remainder = Remainder::new(&t, &dim, &stride);
    let original_bits = total_dim * 8;
    let width_bits = remainder.width_bits();
    // let width = original_bits as f64 * compression_rate / width_bits as f64;
    // let width = width as usize;
    let nbytes = (width * width_bits).div_ceil(8);
    dbg!(width, nbytes);
    let mut cut = Cut::new(&dim, block_size);
    let mut mem = {
        let col_dim = dim.iter().map(|&dim| total_dim / dim).max().unwrap_or(0);

        let x_new = StackReq::new::<u64>(col_dim.div_ceil(64));
        let diff_idx = StackReq::new::<usize>(col_dim);
        let kron = StackReq::new::<u64>(total_dim.div_ceil(64));
        let s_kron = StackReq::new::<u64>(total_dim.div_ceil(64) * block_size);
        let t_kron = StackReq::new::<u64>(total_dim.div_ceil(64) * block_size);
        GlobalPodBuffer::new(StackReq::any_of([
            StackReq::all_of([x_new, diff_idx]),
            StackReq::all_of([s_kron.or(t_kron), s_kron, t_kron]),
            StackReq::all_of([kron, kron]),
        ]))
    };
    let mut stack = PodStack::new(&mut mem);
    let rng = &mut StdRng::seed_from_u64(0);
    let normalization = (3 * nrows * ncols * 255 * 255) as f64;
    let mut approx = Col::zeros(total_dim);
    let mut w = 0;
    while w < width {
        let bs = Ord::min(block_size, width - w);

        remainder.fill_matrices();
        for _ in 0..bs {
            remainder.cut(&mut cut, rng, stack.rb_mut());
        }
        {
            let f = 1.0 / total_dim as f32;
            let scale = &*cut.c().iter().map(|&c| c * f).collect::<Box<[_]>>();
            cut.flush(approx.as_slice_mut(), scale, stack.rb_mut());
        }
        remainder.update(&mut cut, stack.rb_mut());
        cut.reset();

        w += bs;
        {
            // let curr_error = remainder.norm_l2() as f64 / normalization.sqrt();
            let mut curr_err: i64 = 0;
            let out = ImageBuffer::from_fn(nrows as _, ncols as _, |i, j| {
                let i = i as usize;
                let j = j as usize;
                Rgb(core::array::from_fn(|c| {
                    let ijc = c * stride[0] + i * stride[1] + j * stride[2];
                    let c = img.get_pixel(i as _, j as _).0[c];
                    let approx_c = to_u8(approx[ijc]);
                    let err = c as i64 - approx_c as i64;
                    curr_err += err * err;
                    approx_c
                }))
            });
            let curr_err = (curr_err as f64 / normalization).sqrt();
            let nbits = w * (nrows + ncols + 3 + 32);
            println!("({w}, {nbits}, {curr_err}),");
            let outpath = output.join(format!("{stem}-{w:04}.jpg"));
            out.save(outpath)?;
            // dbg!(nbytes);
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
    Ok(())
}

fn to_u8(x: f32) -> u8 {
    assert!(x.is_finite());
    let x = x.clamp(0.0, 255.0);
    x.round() as _
}

fn total_error(a: &[f32], bytes: &[u8]) -> i64 {
    assert!(a.len() == bytes.len());
    let mut err: i64 = 0;
    for (a, b) in a.iter().zip(bytes.iter()) {
        let a = a.clamp(0.0, 255.0).round() as i64;
        let b = *b as i64;
        let e = a - b;
        err += e * e
    }
    err
}

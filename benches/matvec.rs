use aligned_vec::avec;
use cuts::{bitmagic::matvec, MatRef, SignMatRef};
use diol::prelude::*;

fn v4(bencher: Bencher, PlotArg(m): PlotArg) {
    if let Some(simd) = pulp::x86::V4::try_new() {
        let n = m;
        let stride = m.div_ceil(64);
        let data = avec![!0u64; n * stride];
        let lhs =
            SignMatRef::from_storage(MatRef::from_col_major_slice(&data, stride, n, stride), m);

        let rhs = &*avec![1.0; n];
        let dst = &mut *avec![1.0; m];

        bencher.bench(|| matvec::x86::matvec_f32_v4(simd, dst, lhs, rhs))
    }
}

fn v3(bencher: Bencher, PlotArg(m): PlotArg) {
    if let Some(simd) = pulp::x86::V3::try_new() {
        let n = m;
        let stride = m.div_ceil(64);
        let data = avec![!0u64; n * stride];
        let lhs =
            SignMatRef::from_storage(MatRef::from_col_major_slice(&data, stride, n, stride), m);

        let rhs = &*avec![1.0; n];
        let dst = &mut *avec![1.0; m];

        bencher.bench(|| matvec::x86::matvec_f32_v3(simd, dst, lhs, rhs))
    }
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(list![v3, v4], [32, 64, 256, 1024, 4096].map(PlotArg));
    bench.run()?;

    Ok(())
}

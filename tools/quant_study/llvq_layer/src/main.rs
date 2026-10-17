//! Quantize one weight matrix with llvq's Leech-lattice Spherical GPTQ (Tetra shape-gain, 48-bit words per 24
//! weights) for the GLM-5.3-Flash quantization study (docs/GLM_QUANT_LEECH_GSQ_STUDY.md).
//!
//!   llvq_layer ROWS COLS in.f32 out.f32 [--hdiag diag.f32 | --hfull h.f32] [--rot SEED] [--threads N]
//!
//! Without a Hessian the layer is quantized with H = I (data-free). `--hdiag` takes per-column E[x^2] (the engine's
//! calibration moments), `--hfull` a COLS x COLS row-major f32 matrix. `--rot` applies the incoherence rotation of
//! llvq_quant::rotation (W' = W Q^T, H' = Q H Q^T, then rotates the result back). Rows are independent given the
//! shared GPTQ factor, so they are split across threads. Prints one JSON line with the bits per weight.
use llvq_quant::gptq::{bits_per_weight, quantize_layer, GptqConfig, TailPolicy, Weights};
use llvq_quant::linalg::GptqFactor;
use llvq_quant::quantizer::{fit_gain_centroids, TetraShapeGain};
use llvq_quant::rotation::Rotation;
use std::io::{Read, Write};

fn read_f32(path: &str, n: usize) -> Vec<f64> {
    let mut bytes = Vec::new();
    std::fs::File::open(path).and_then(|mut f| f.read_to_end(&mut bytes)).expect("read input");
    assert_eq!(bytes.len(), n * 4, "{path}: expected {n} floats");
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: llvq_layer ROWS COLS in.f32 out.f32 [--hdiag f] [--hfull f] [--rot SEED] [--threads N]");
        std::process::exit(2);
    }
    let rows: usize = args[1].parse().unwrap();
    let cols: usize = args[2].parse().unwrap();
    let mut w = read_f32(&args[3], rows * cols);
    let (mut hdiag, mut hfull, mut rot, mut threads) = (None, None, None, 8usize);
    let mut i = 5;
    while i < args.len() {
        match args[i].as_str() {
            "--hdiag" => { hdiag = Some(args[i + 1].clone()); i += 2; }
            "--hfull" => { hfull = Some(args[i + 1].clone()); i += 2; }
            "--rot" => { rot = Some(args[i + 1].parse::<u64>().unwrap()); i += 2; }
            "--threads" => { threads = args[i + 1].parse().unwrap(); i += 2; }
            other => panic!("unknown option {other}"),
        }
    }
    // Hessian: identity, diagonal or full.
    let mut h = vec![0.0f64; cols * cols];
    match (&hdiag, &hfull) {
        (Some(p), _) => { let d = read_f32(p, cols); for k in 0..cols { h[k * cols + k] = d[k].max(1e-12); } }
        (None, Some(p)) => { h = read_f32(p, cols * cols); }
        (None, None) => { for k in 0..cols { h[k * cols + k] = 1.0; } }
    }
    let rotation = rot.map(|seed| Rotation::new(cols, seed));
    if let Some(q) = &rotation {
        q.rotate_weight_rows(&mut w, rows);
        q.rotate_hessian(&mut h);
    }
    let t0 = std::time::Instant::now();
    let factor = GptqFactor::new(&h, cols, 0.01).expect("Hessian not positive definite after damping");
    let block = 24;
    let centroids = fit_gain_centroids(&w, rows, cols, block, 1, 20);
    let cfg = GptqConfig { tail: TailPolicy::KeepExact, ..GptqConfig::default() };
    let encoder = TetraShapeGain::encoder();
    let per = (rows + threads - 1) / threads;
    let chunks: Vec<Vec<f64>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .filter(|t| t * per < rows)
            .map(|t| {
                let (lo, hi) = (t * per, ((t + 1) * per).min(rows));
                let part = w[lo * cols..hi * cols].to_vec();
                let (factor, cfg, enc, cents) = (&factor, &cfg, encoder.clone(), centroids.clone());
                s.spawn(move || {
                    let mut weights = Weights::new(hi - lo, cols, part);
                    let mut q = TetraShapeGain::with_encoder(enc, cents);
                    quantize_layer(&mut weights, factor, None, &mut q, cfg);
                    weights.w
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut out: Vec<f64> = chunks.into_iter().flatten().collect();
    if let Some(q) = &rotation {
        q.unrotate_weight_rows(&mut out, rows);
    }
    let mut f = std::fs::File::create(&args[4]).expect("create output");
    let bytes: Vec<u8> = out.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect();
    f.write_all(&bytes).expect("write output");
    // 48 bits per 24-weight block plus a 16-bit row scale; the tail stays at 16 bits (BF16).
    let bpw = bits_per_weight(cols, block, 48, 16) + 16.0 / cols as f64;
    println!("{{\"method\":\"llvq-tetra\",\"hessian\":\"{}\",\"rotation\":{},\"bpw\":{:.4},\"seconds\":{:.1},\"centroids\":{:?}}}",
             if hdiag.is_some() { "diag" } else if hfull.is_some() { "full" } else { "identity" },
             rot.map(|s| s.to_string()).unwrap_or("null".into()), bpw, t0.elapsed().as_secs_f64(), centroids);
}

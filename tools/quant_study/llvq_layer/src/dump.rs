//! Reference vectors for the C++ Tetra port (tests/llvq): N blocks of 24 SplitMix64 Gaussians, scaled per block by
//! 10^u (u uniform in [-3, 3]) with special blocks (zero, tiny, huge, sparse) and two thirds rounded to a coarse grid
//! (exact ties), and the Rust encoder's word and
//! t for each. Output: little-endian records of 24 f64 (block) + u64 word + f64 t.
//!   llvq_dump N SEED out.bin
//! Matrix mode, the C++ tool's per-matrix pipeline with an identity Hessian (rotate rows, fit two gain centroids,
//! TetraShapeGain on every full block, tail kept): centroids (2 f64), row scales (ROWS f64), then
//! ROWS x (COLS/24) u64 words with the gain level at bit 47.
//!   llvq_dump matrix ROWS COLS in.f32 ROT_SEED ITERS out.bin
use llvq_core::SplitMix64;
use llvq_quant::quantizer::{fit_gain_centroids, row_scale, BlockQuantizer, TetraShapeGain};
use llvq_quant::rotation::Rotation;
use llvq_search::tetra::{Encoder, Scratch, Tetra};
use std::io::Write;

fn matrix(a: &[String]) {
    let rows: usize = a[2].parse().unwrap();
    let cols: usize = a[3].parse().unwrap();
    let bytes = std::fs::read(&a[4]).unwrap();
    assert_eq!(bytes.len(), rows * cols * 4);
    let mut w: Vec<f64> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64).collect();
    let seed: u64 = a[5].parse().unwrap();
    let iters: usize = a[6].parse().unwrap();
    Rotation::new(cols, seed).rotate_weight_rows(&mut w, rows);
    let cents = fit_gain_centroids(&w, rows, cols, 24, 1, iters);
    let tetra = Tetra::new();
    let mut q = TetraShapeGain::with_encoder(TetraShapeGain::encoder(), cents.clone());
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[7]).unwrap());
    for c in &cents {
        out.write_all(&c.to_le_bytes()).unwrap();
    }
    let scales: Vec<f64> = (0..rows).map(|i| row_scale(&w[i * cols..(i + 1) * cols])).collect();
    for s in &scales {
        out.write_all(&s.to_le_bytes()).unwrap();
    }
    let mut buf = [0f64; 24];
    for i in 0..rows {
        q.set_row_scale(scales[i]);
        for b in 0..cols / 24 {
            q.quantize(&w[i * cols + 24 * b..i * cols + 24 * b + 24], &mut buf);
            let code = q.last_code().unwrap();
            let word = tetra.encode(&code.point).expect("a Tetra point") | (code.gain as u64) << 47;
            out.write_all(&word.to_le_bytes()).unwrap();
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "matrix" {
        return matrix(&a);
    }
    let n: usize = a[1].parse().unwrap();
    let seed: u64 = u64::from_str_radix(a[2].trim_start_matches("0x"), 16).unwrap();
    let tetra = Tetra::new();
    let enc = Encoder::new(&tetra);
    let mut sc = Scratch::new();
    let mut rng = SplitMix64::new(seed);
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[3]).unwrap());
    for i in 0..n {
        let mut x = [0f64; 24];
        let scale = 10f64.powf(rng.next_f64() * 6.0 - 3.0);
        for v in x.iter_mut() {
            *v = rng.next_gaussian() * scale;
        }
        match i % 97 {
            0 => x = [0.0; 24],
            1 => x.iter_mut().for_each(|v| *v *= 1e-300),
            2 => x.iter_mut().for_each(|v| *v *= 1e250),
            3 => x.iter_mut().enumerate().for_each(|(j, v)| if j % 5 != 0 { *v = 0.0 }),
            // FP8-like discrete values: many exact ties between candidate distances.
            k if k % 3 == 0 => x.iter_mut().for_each(|v| *v = (*v / scale * 4.0).round() / 4.0 * scale),
            k if k % 3 == 1 => x.iter_mut().for_each(|v| *v = (*v / scale).round()),
            _ => {}
        }
        let c = enc.encode(&x, &mut sc);
        for v in x {
            out.write_all(&v.to_le_bytes()).unwrap();
        }
        out.write_all(&c.word.to_le_bytes()).unwrap();
        out.write_all(&c.t.to_le_bytes()).unwrap();
    }
}

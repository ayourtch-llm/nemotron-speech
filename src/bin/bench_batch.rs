//! Micro-benchmark: does batching two streams into one encoder forward pay off?
//! Compares 2× batch-1 forwards (today's tag_live: two separate pipelines)
//! against 1× batch-2 forward (option 3) for the FastConformer encoder, on
//! whichever device the build/flags select. Prints per-call ms and the speedup.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use nemotron_speech::model::encoder::FastConformerEncoder;
use nemotron_speech::model::ModelConfig;

fn main() -> Result<()> {
    let mut st = String::new();
    let mut cpu = false;
    let mut frames = 112usize; // ~1.12s of mel (one streaming chunk's worth)
    let mut iters = 30usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--st" => st = args.next().unwrap(),
            "--cpu" => cpu = true,
            "--frames" => frames = args.next().unwrap().parse()?,
            "--iters" => iters = args.next().unwrap().parse()?,
            o => anyhow::bail!("unknown arg {o}"),
        }
    }
    let device = if cpu {
        Device::Cpu
    } else {
        #[cfg(feature = "metal")]
        { Device::new_metal(0).unwrap_or(Device::Cpu) }
        #[cfg(not(feature = "metal"))]
        { Device::Cpu }
    };
    let dtype = DType::F32;
    eprintln!("device {:?}, frames {frames} (~{:.2}s), iters {iters}", device, frames as f32 / 100.0);

    let cfg = ModelConfig::nemotron_06b();
    let st_path = std::path::PathBuf::from(&st);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[st_path], dtype, &device).context("st")? };
    let encoder = FastConformerEncoder::new(vb.pp("encoder"), cfg.clone())
        .map_err(|e| anyhow::anyhow!("encoder: {e:#}"))?;

    let n_mels = 128;
    let mel1 = Tensor::randn(0f32, 1.0, (1, n_mels, frames), &device)?;
    let mel2 = Tensor::randn(0f32, 1.0, (2, n_mels, frames), &device)?;

    // Warmup (and force any lazy device init / kernel compile).
    for _ in 0..3 {
        let _ = encoder.forward_full(&mel1, false)?.to_vec3::<f32>().ok();
        let _ = encoder.forward_full(&mel2, false)?.to_vec3::<f32>().ok();
    }

    // 2× batch-1 (two separate streams, today's approach).
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let a = encoder.forward_full(&mel1, false)?;
        let b = encoder.forward_full(&mel1, false)?;
        let _ = (a.to_vec3::<f32>()?, b.to_vec3::<f32>()?); // sync device
    }
    let two_b1 = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // 1× batch-2 (option 3).
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let a = encoder.forward_full(&mel2, false)?;
        let _ = a.to_vec3::<f32>()?; // sync device
    }
    let one_b2 = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // 2× batch-1 on TWO THREADS (the parallelism-across-cores hypothesis).
    let enc = std::sync::Arc::new(encoder);
    let two_par = {
        let t0 = std::time::Instant::now();
        let mut handles = Vec::new();
        for _ in 0..2 {
            let enc = enc.clone();
            let mel = mel1.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..iters {
                    let a = enc.forward_full(&mel, false).unwrap();
                    let _ = a.to_vec3::<f32>().unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    };

    println!("2× batch-1 (sequential): {two_b1:.1} ms/iter");
    println!("2× batch-1 (2 threads):  {two_par:.1} ms/iter   <- parallelism across cores");
    println!("1× batch-2 (shared):     {one_b2:.1} ms/iter");
    println!("threading speedup: {:.2}×   batching speedup: {:.2}×", two_b1 / two_par, two_b1 / one_b2);
    let chunk_s = frames as f64 / 100.0;
    println!(
        "batch-2 throughput: {:.1}× realtime for TWO streams ({:.0} ms compute per {:.2}s of audio×2)",
        (chunk_s * 2.0 * 1000.0) / one_b2,
        one_b2,
        chunk_s
    );
    Ok(())
}

use anyhow::Result;
use std::fs;
use std::io::Read;
use std::time::Instant;

use rust_nn_cnn::NeuralNetwork;

fn read_idx_images(path: &str) -> Result<Vec<f32>> {
    let mut data = Vec::new();
    fs::File::open(path)?.read_to_end(&mut data)?;
    if data.len() < 16 {
        anyhow::bail!("bad idx image header");
    }
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 2051 {
        anyhow::bail!("bad image magic");
    }
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let mut out = Vec::with_capacity(n * rows * cols);
    for i in 0..n * rows * cols {
        out.push(data[16 + i] as f32 / 255.0);
    }
    Ok(out)
}

fn read_idx_labels(path: &str) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    fs::File::open(path)?.read_to_end(&mut data)?;
    if data.len() < 8 {
        anyhow::bail!("bad idx label header");
    }
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 2049 {
        anyhow::bail!("bad label magic");
    }
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(data[8 + i]);
    }
    Ok(out)
}

fn main() -> Result<()> {
    env_logger::init();

    let images = read_idx_images("dataset/train-images-idx3-ubyte")?;
    let labels = read_idx_labels("dataset/train-labels-idx1-ubyte")?;

    // Build dataset slices
    let num = labels.len().min(10_000);
    let in_dim = 28 * 28;
    let out_dim = 10;
    let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(num);
    let mut targets: Vec<Vec<f32>> = Vec::with_capacity(num);

    for i in 0..num {
        let start = i * in_dim;
        inputs.push(images[start..start + in_dim].to_vec());

        let mut oh = vec![0f32; out_dim];
        oh[labels[i] as usize] = 1.0;
        targets.push(oh);
    }

    // Create or load model (resume if checkpoint exists)
    std::fs::create_dir_all("models").ok();
    let ckpt_base = "models/mnist_1dcnn"; // Recorder appends .mpk
    let mut nn = match NeuralNetwork::load(ckpt_base) {
        Ok(model) => {
            println!("loading checkpoint: {}.mpk", ckpt_base);
            model
        }
        Err(e) => {
            println!(
                "no checkpoint found or failed to load ({}), starting fresh",
                e
            );
            NeuralNetwork::new(&[in_dim, 256, 256, out_dim], 128)
        }
    }
    .with_learning_rate(1e-3);

    // Train until loss is below threshold
    let mut epoch = 0usize;
    loop {
        let t0 = Instant::now();
        epoch += 1;

        let avg_loss = nn.train_batch(&inputs, &targets);
        let dt = t0.elapsed();

        println!(
            "epoch {} avg_loss {:.4} time {:.3}s",
            epoch,
            avg_loss,
            dt.as_secs_f64()
        );

        // Save checkpoint if model improved
        if let Ok(true) = nn.try_save_checkpoint(ckpt_base, avg_loss) {
            println!("checkpoint saved: {}.mpk (loss {:.4})", ckpt_base, avg_loss);
        }
        if avg_loss < 0.001 {
            break;
        }
    }

    Ok(())
}

//! Benchmark GPU vs CPU distance computation
//!
//! Run with: cargo run --release --features gpu --bin benchmark_gpu

fn main() {
    #[cfg(feature = "gpu")]
    {
        run_gpu_benchmark();
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled. Run with: cargo run --release --features gpu --bin benchmark_gpu");
        println!("\nRunning CPU-only benchmark...");
        run_cpu_benchmark();
    }
}

#[cfg(feature = "gpu")]
fn run_gpu_benchmark() {
    use pardusdb::gpu::GpuDistance;
    use std::time::Instant;

    println!("=== GPU vs CPU Distance Benchmark ===\n");

    // Initialize GPU
    println!("Initializing GPU...");
    let gpu = pollster::block_on(GpuDistance::new())
        .expect("Failed to initialize GPU");

    println!("GPU initialized successfully!\n");

    const NUM_VECTORS: usize = 1000;
    const DIM: usize = 768;

    // Generate test data
    println!("Generating {} vectors (dim={})...", NUM_VECTORS, DIM);
    let query: Vec<f32> = (0..DIM).map(|i| (i as f32 / DIM as f32)).collect();
    let vectors: Vec<Vec<f32>> = (0..NUM_VECTORS)
        .map(|i| (0..DIM).map(|j| ((i * DIM + j) as f32 / (NUM_VECTORS * DIM) as f32)).collect())
        .collect();
    println!("Generated in {:?}", {
        let start = Instant::now();
        let _: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        start.elapsed()
    });

    // Warm up GPU
    println!("\nWarming up GPU...");
    let _ = gpu.euclidean_batch(&query, &vectors[0..10]);

    // GPU benchmark
    println!("\n--- GPU Benchmark ---");
    let mut gpu_total = std::time::Duration::ZERO;
    let iterations = 10;

    for i in 0..iterations {
        let start = Instant::now();
        let distances = gpu.euclidean_batch(&query, &vectors).unwrap();
        gpu_total += start.elapsed();

        if i == 0 {
            println!("Sample distances: {:.4}, {:.4}, {:.4}",
                distances[0], distances[NUM_VECTORS/2], distances[NUM_VECTORS-1]);
        }
    }

    println!("GPU: {} iterations in {:?}", iterations, gpu_total);
    println!("GPU avg: {:?} per batch ({:?} per vector)",
        gpu_total / iterations as u32,
        gpu_total / (iterations * NUM_VECTORS) as u32);

    // CPU benchmark
    println!("\n--- CPU Benchmark ---");
    let mut cpu_total = std::time::Duration::ZERO;

    for i in 0..iterations {
        let start = Instant::now();
        let distances: Vec<f32> = vectors.iter()
            .map(|v| {
                let sum: f32 = query.iter().zip(v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                sum.sqrt()
            })
            .collect();
        cpu_total += start.elapsed();

        if i == 0 {
            println!("Sample distances: {:.4}, {:.4}, {:.4}",
                distances[0], distances[NUM_VECTORS/2], distances[NUM_VECTORS-1]);
        }
    }

    println!("CPU: {} iterations in {:?}", iterations, cpu_total);
    println!("CPU avg: {:?} per batch ({:?} per vector)",
        cpu_total / iterations as u32,
        cpu_total / (iterations * NUM_VECTORS) as u32);

    // Summary
    println!("\n=== Summary ===");
    println!("Vectors: {}", NUM_VECTORS);
    println!("Dimension: {}", DIM);
    let speedup = cpu_total.as_secs_f64() / gpu_total.as_secs_f64();
    println!("GPU speedup: {:.1}x", speedup);

    if speedup > 1.0 {
        println!("GPU is {:.1}x faster than CPU", speedup);
    } else {
        println!("GPU is {:.1}x slower than CPU (overhead may dominate for small batches)", 1.0/speedup);
    }
}

#[cfg(not(feature = "gpu"))]
fn run_cpu_benchmark() {
    use std::time::Instant;

    const NUM_VECTORS: usize = 1000;
    const DIM: usize = 768;

    println!("=== CPU Distance Benchmark ===\n");

    // Generate test data
    println!("Generating {} vectors (dim={})...", NUM_VECTORS, DIM);
    let query: Vec<f32> = (0..DIM).map(|i| (i as f32 / DIM as f32)).collect();
    let vectors: Vec<Vec<f32>> = (0..NUM_VECTORS)
        .map(|i| (0..DIM).map(|j| ((i * DIM + j) as f32 / (NUM_VECTORS * DIM) as f32)).collect())
        .collect();

    // CPU benchmark
    println!("\n--- CPU Benchmark ---");
    let mut cpu_total = std::time::Duration::ZERO;
    let iterations = 10;

    for i in 0..iterations {
        let start = Instant::now();
        let distances: Vec<f32> = vectors.iter()
            .map(|v| {
                let sum: f32 = query.iter().zip(v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                sum.sqrt()
            })
            .collect();
        cpu_total += start.elapsed();

        if i == 0 {
            println!("Sample distances: {:.4}, {:.4}, {:.4}",
                distances[0], distances[NUM_VECTORS/2], distances[NUM_VECTORS-1]);
        }
    }

    println!("CPU: {} iterations in {:?}", iterations, cpu_total);
    println!("CPU avg: {:?} per batch ({:?} per vector)",
        cpu_total / iterations as u32,
        cpu_total / (iterations * NUM_VECTORS) as u32);

    println!("\nNote: Enable GPU feature for hardware acceleration:");
    println!("  cargo run --release --features gpu --bin benchmark_gpu");
}

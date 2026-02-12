//! Benchmark GPU vs CPU with larger batches
//!
//! Run with: cargo run --release --features gpu --bin benchmark_gpu_scaled

fn main() {
    #[cfg(feature = "gpu")]
    {
        use pardusdb::gpu::GpuDistance;
        use std::time::Instant;

        println!("=== GPU vs CPU Distance Benchmark (Scaled) ===\n");

        // Initialize GPU
        println!("Initializing GPU...");
        let gpu = pollster::block_on(GpuDistance::new())
            .expect("Failed to initialize GPU");
        println!("GPU initialized successfully!\n");

        // Test different scales
        let configs = [
            (100, 768),
            (1000, 768),
            (10000, 768),
            (1000, 1536),  // Higher dimension
        ];

        for (num_vectors, dim) in configs {
            println!("--- {} vectors, dim={} ---", num_vectors, dim);

            // Generate test data
            let query: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
            let vectors: Vec<Vec<f32>> = (0..num_vectors)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 / (num_vectors * dim) as f32)).collect())
                .collect();

            // Warm up
            let _ = gpu.euclidean_batch(&query, &vectors[0..10.min(num_vectors)]);

            // GPU benchmark
            let iterations = 5;
            let mut gpu_total = std::time::Duration::ZERO;

            for _ in 0..iterations {
                let start = Instant::now();
                let _ = gpu.euclidean_batch(&query, &vectors).unwrap();
                gpu_total += start.elapsed();
            }

            // CPU benchmark - use black_box to prevent optimization
            let mut cpu_total = std::time::Duration::ZERO;
            let mut sink = 0.0f32;

            for _ in 0..iterations {
                let start = Instant::now();
                let distances: Vec<f32> = vectors.iter()
                    .map(|v| {
                        let sum: f32 = query.iter().zip(v.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        sum.sqrt()
                    })
                    .collect();
                // Prevent optimization by using the result
                sink += distances.iter().sum::<f32>();
                cpu_total += start.elapsed();
            }
            // Final use of sink
            println!("  (checksum: {:.4})", sink);

            let gpu_avg = gpu_total / iterations as u32;
            let cpu_avg = cpu_total / iterations as u32;
            let speedup = cpu_avg.as_secs_f64() / gpu_avg.as_secs_f64();

            println!("  GPU: {:?} ({:.2?}/vec)", gpu_avg, gpu_avg / num_vectors as u32);
            println!("  CPU: {:?} ({:.2?}/vec)", cpu_avg, cpu_avg / num_vectors as u32);

            if speedup > 1.0 {
                println!("  GPU is {:.1}x FASTER", speedup);
            } else {
                println!("  GPU is {:.1}x slower (overhead dominates)", 1.0/speedup);
            }
            println!();
        }

        println!("=== Analysis ===");
        println!("GPU excels with:");
        println!("  - Larger batches (10,000+ vectors)");
        println!("  - Higher dimensions (1536+ for embeddings)");
        println!("  - Multiple queries processed together");
        println!("\nCPU is faster for small batches due to GPU overhead:");
        println!("  - Buffer allocation");
        println!("  - Data transfer (PCIe bus)");
        println!("  - Kernel dispatch");
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled.");
        println!("Run with: cargo run --release --features gpu --bin benchmark_gpu_scaled");
    }
}

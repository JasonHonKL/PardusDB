//! GPU-accelerated distance computation using wgpu
//!
//! This module provides GPU-accelerated distance computation for vector similarity search.
//! It uses compute shaders to parallelize distance calculations across GPU cores.
//!
//! Supported platforms:
//! - macOS: Metal
//! - Windows: DirectX 12
//! - Linux: Vulkan

#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu")]
use std::sync::Arc;
#[cfg(feature = "gpu")]
use wgpu::*;
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// GPU distance computation context
#[cfg(feature = "gpu")]
pub struct GpuDistance {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline_euclidean: ComputePipeline,
    pipeline_cosine: ComputePipeline,
    pipeline_dot: ComputePipeline,
}

#[cfg(feature = "gpu")]
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct ComputeParams {
    dimension: u32,
    num_vectors: u32,
    _padding: [u32; 2],
}

#[cfg(feature = "gpu")]
impl GpuDistance {
    /// Create a new GPU distance context
    pub async fn new() -> Result<Self, GpuError> {
        // Initialize wgpu
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("PardusDB GPU Device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: MemoryHints::default(),
            }, None)
            .await
            .map_err(|e| GpuError::DeviceError(e.to_string()))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create compute pipelines
        let pipeline_euclidean = Self::create_euclidean_pipeline(&device)?;
        let pipeline_cosine = Self::create_cosine_pipeline(&device)?;
        let pipeline_dot = Self::create_dot_pipeline(&device)?;

        Ok(GpuDistance {
            device,
            queue,
            pipeline_euclidean,
            pipeline_cosine,
            pipeline_dot,
        })
    }

    fn create_euclidean_pipeline(device: &Device) -> Result<ComputePipeline, GpuError> {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Euclidean Distance Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/euclidean.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Euclidean Bind Group Layout"),
            entries: &[
                // Query vector (read-only storage)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Vectors (read-only storage)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Results (read-write storage)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params (uniform)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Euclidean Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        Ok(device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Euclidean Distance Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }))
    }

    fn create_cosine_pipeline(device: &Device) -> Result<ComputePipeline, GpuError> {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Cosine Distance Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/cosine.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Cosine Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Cosine Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        Ok(device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Cosine Distance Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }))
    }

    fn create_dot_pipeline(device: &Device) -> Result<ComputePipeline, GpuError> {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Dot Product Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/dot.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Dot Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Dot Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        Ok(device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Dot Product Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }))
    }

    /// Compute Euclidean distances from query to multiple vectors
    pub fn euclidean_batch(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>, GpuError> {
        self.compute_distances(query, vectors, &self.pipeline_euclidean)
    }

    /// Compute Cosine distances from query to multiple vectors
    pub fn cosine_batch(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>, GpuError> {
        self.compute_distances(query, vectors, &self.pipeline_cosine)
    }

    /// Compute Dot products from query to multiple vectors
    pub fn dot_batch(&self, query: &[f32], vectors: &[Vec<f32>]) -> Result<Vec<f32>, GpuError> {
        self.compute_distances(query, vectors, &self.pipeline_dot)
    }

    fn compute_distances(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        pipeline: &ComputePipeline,
    ) -> Result<Vec<f32>, GpuError> {
        let dimension = query.len();
        let num_vectors = vectors.len();

        if num_vectors == 0 {
            return Ok(Vec::new());
        }

        // Flatten vectors into contiguous array
        let mut flat_vectors = Vec::with_capacity(dimension * num_vectors);
        for vec in vectors {
            if vec.len() != dimension {
                return Err(GpuError::DimensionMismatch);
            }
            flat_vectors.extend_from_slice(vec);
        }

        // Create buffers
        let query_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Query Buffer"),
            contents: bytemuck::cast_slice(query),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let vectors_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Vectors Buffer"),
            contents: bytemuck::cast_slice(&flat_vectors),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let results_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Results Buffer"),
            size: (num_vectors * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (num_vectors * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = ComputeParams {
            dimension: dimension as u32,
            num_vectors: num_vectors as u32,
            _padding: [0, 0],
        };
        let params_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Distance Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: vectors_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Distance Compute Encoder"),
        });

        // Dispatch compute
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Distance Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_vectors as u32, 1, 1);
        }

        // Copy results to staging buffer
        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            (num_vectors * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv().unwrap().map_err(|e| GpuError::MappingError(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

/// GPU-related errors
#[derive(Debug, Clone)]
pub enum GpuError {
    /// No suitable GPU adapter found
    NoAdapter,
    /// Failed to create device
    DeviceError(String),
    /// Failed to map buffer
    MappingError(String),
    /// Vector dimension mismatch
    DimensionMismatch,
    /// Shader compilation error
    ShaderError(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "No suitable GPU adapter found"),
            GpuError::DeviceError(e) => write!(f, "GPU device error: {}", e),
            GpuError::MappingError(e) => write!(f, "GPU buffer mapping error: {}", e),
            GpuError::DimensionMismatch => write!(f, "Vector dimension mismatch"),
            GpuError::ShaderError(e) => write!(f, "Shader error: {}", e),
        }
    }
}

impl std::error::Error for GpuError {}

// Fallback CPU implementation when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
pub struct GpuDistance;

#[cfg(not(feature = "gpu"))]
impl GpuDistance {
    pub async fn new() -> Result<Self, GpuError> {
        Err(GpuError::NoAdapter)
    }

    pub fn euclidean_batch(&self, _query: &[f32], _vectors: &[Vec<f32>]) -> Result<Vec<f32>, GpuError> {
        Err(GpuError::NoAdapter)
    }
}

#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone)]
pub enum GpuError {
    NoAdapter,
    DeviceError(String),
    MappingError(String),
    DimensionMismatch,
    ShaderError(String),
}

#[cfg(not(feature = "gpu"))]
impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "GPU support not enabled - compile with 'gpu' feature"),
            GpuError::DeviceError(e) => write!(f, "GPU device error: {}", e),
            GpuError::MappingError(e) => write!(f, "GPU buffer mapping error: {}", e),
            GpuError::DimensionMismatch => write!(f, "Vector dimension mismatch"),
            GpuError::ShaderError(e) => write!(f, "Shader error: {}", e),
        }
    }
}

#[cfg(not(feature = "gpu"))]
impl std::error::Error for GpuError {}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_euclidean() {
        pollster::block_on(async {
            let gpu = GpuDistance::new().await.unwrap();

            let query = vec![1.0, 0.0, 0.0];
            let vectors = vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ];

            let distances = gpu.euclidean_batch(&query, &vectors).unwrap();

            assert!((distances[0] - 0.0).abs() < 0.001);
            assert!((distances[1] - 1.4142).abs() < 0.01);
            assert!((distances[2] - 1.4142).abs() < 0.01);
        });
    }
}

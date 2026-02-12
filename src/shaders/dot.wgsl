// Dot product compute shader
// Computes negative dot product (for max-heap style nearest neighbor search)

struct ComputeParams {
    dimension: u32,
    num_vectors: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> vectors: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;
@group(0) @binding(3) var<uniform> params: ComputeParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vector_idx = global_id.x;

    if (vector_idx >= params.num_vectors) {
        return;
    }

    let start_idx = vector_idx * params.dimension;
    var dot_product: f32 = 0.0;

    for (var i: u32 = 0u; i < params.dimension; i++) {
        dot_product += query[i] * vectors[start_idx + i];
    }

    // Return negative dot product (higher dot = smaller distance for NN search)
    results[vector_idx] = -dot_product;
}

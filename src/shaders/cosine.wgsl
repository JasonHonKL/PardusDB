// Cosine distance compute shader
// Computes 1 - cosine_similarity between query vector and each vector in the batch

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
    var query_norm: f32 = 0.0;
    var vec_norm: f32 = 0.0;

    for (var i: u32 = 0u; i < params.dimension; i++) {
        let q = query[i];
        let v = vectors[start_idx + i];
        dot_product += q * v;
        query_norm += q * q;
        vec_norm += v * v;
    }

    let norm_product = sqrt(query_norm) * sqrt(vec_norm);
    var cosine_sim: f32 = 0.0;

    if (norm_product > 0.0) {
        cosine_sim = dot_product / norm_product;
    }

    // Return cosine distance (1 - similarity)
    results[vector_idx] = 1.0 - cosine_sim;
}

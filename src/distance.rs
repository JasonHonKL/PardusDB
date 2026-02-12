/// Numeric trait for vector elements.
/// Supports both floating point and integer types for flexibility.
pub trait Numeric: Copy + Clone + Send + Sync + 'static {
    fn to_f32(self) -> f32;
    fn zero() -> Self;
}

impl Numeric for f32 {
    fn to_f32(self) -> f32 { self }
    fn zero() -> Self { 0.0 }
}

impl Numeric for f64 {
    fn to_f32(self) -> f32 { self as f32 }
    fn zero() -> Self { 0.0 }
}

impl Numeric for i32 {
    fn to_f32(self) -> f32 { self as f32 }
    fn zero() -> Self { 0 }
}

impl Numeric for i64 {
    fn to_f32(self) -> f32 { self as f32 }
    fn zero() -> Self { 0 }
}

impl Numeric for u32 {
    fn to_f32(self) -> f32 { self as f32 }
    fn zero() -> Self { 0 }
}

impl Numeric for u64 {
    fn to_f32(self) -> f32 { self as f32 }
    fn zero() -> Self { 0 }
}

/// Distance metric trait for vector similarity.
/// Generic over the numeric type for both float and integer vectors.
pub trait Distance<T: Numeric>: Send + Sync {
    fn compute(a: &[T], b: &[T]) -> f32;
}

// Unroll factor for optimized loops
const UNROLL: usize = 8;

/// Optimized cosine distance for f32 vectors.
/// Uses loop unrolling for better auto-vectorization.
#[inline]
fn cosine_f32_optimized(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    if len < UNROLL {
        // Handle small vectors directly
        let (dot, norm_a, norm_b) = a.iter()
            .zip(b.iter())
            .fold((0.0f32, 0.0f32, 0.0f32), |(d, na, nb), (&x, &y)| {
                (d + x * y, na + x * x, nb + y * y)
            });
        let denom = norm_a.sqrt() * norm_b.sqrt();
        return if denom == 0.0 { 0.0 } else { 1.0 - (dot / denom) };
    }

    let unrolled_len = len - (len % UNROLL);

    // Unrolled loop for main body - multiple accumulators for better ILP
    let mut dot0 = 0.0f32;
    let mut dot1 = 0.0f32;
    let mut dot2 = 0.0f32;
    let mut dot3 = 0.0f32;
    let mut norm_a0 = 0.0f32;
    let mut norm_a1 = 0.0f32;
    let mut norm_a2 = 0.0f32;
    let mut norm_a3 = 0.0f32;
    let mut norm_b0 = 0.0f32;
    let mut norm_b1 = 0.0f32;
    let mut norm_b2 = 0.0f32;
    let mut norm_b3 = 0.0f32;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // Process 8 elements at a time
    for i in (0..unrolled_len).step_by(UNROLL) {
        let xa0 = unsafe { *a_ptr.add(i) };
        let xb0 = unsafe { *b_ptr.add(i) };
        dot0 += xa0 * xb0;
        norm_a0 += xa0 * xa0;
        norm_b0 += xb0 * xb0;

        let xa1 = unsafe { *a_ptr.add(i + 1) };
        let xb1 = unsafe { *b_ptr.add(i + 1) };
        dot1 += xa1 * xb1;
        norm_a1 += xa1 * xa1;
        norm_b1 += xb1 * xb1;

        let xa2 = unsafe { *a_ptr.add(i + 2) };
        let xb2 = unsafe { *b_ptr.add(i + 2) };
        dot2 += xa2 * xb2;
        norm_a2 += xa2 * xa2;
        norm_b2 += xb2 * xb2;

        let xa3 = unsafe { *a_ptr.add(i + 3) };
        let xb3 = unsafe { *b_ptr.add(i + 3) };
        dot3 += xa3 * xb3;
        norm_a3 += xa3 * xa3;
        norm_b3 += xb3 * xb3;

        let xa4 = unsafe { *a_ptr.add(i + 4) };
        let xb4 = unsafe { *b_ptr.add(i + 4) };
        dot0 += xa4 * xb4;
        norm_a0 += xa4 * xa4;
        norm_b0 += xb4 * xb4;

        let xa5 = unsafe { *a_ptr.add(i + 5) };
        let xb5 = unsafe { *b_ptr.add(i + 5) };
        dot1 += xa5 * xb5;
        norm_a1 += xa5 * xa5;
        norm_b1 += xb5 * xb5;

        let xa6 = unsafe { *a_ptr.add(i + 6) };
        let xb6 = unsafe { *b_ptr.add(i + 6) };
        dot2 += xa6 * xb6;
        norm_a2 += xa6 * xa6;
        norm_b2 += xb6 * xb6;

        let xa7 = unsafe { *a_ptr.add(i + 7) };
        let xb7 = unsafe { *b_ptr.add(i + 7) };
        dot3 += xa7 * xb7;
        norm_a3 += xa7 * xa7;
        norm_b3 += xb7 * xb7;
    }

    // Combine accumulators
    let dot = dot0 + dot1 + dot2 + dot3;
    let norm_a = norm_a0 + norm_a1 + norm_a2 + norm_a3;
    let norm_b = norm_b0 + norm_b1 + norm_b2 + norm_b3;

    // Handle remaining elements
    let (dot_final, norm_a_final, norm_b_final) = a[unrolled_len..]
        .iter()
        .zip(b[unrolled_len..].iter())
        .fold((dot, norm_a, norm_b), |(d, na, nb), (&x, &y)| {
            (d + x * y, na + x * x, nb + y * y)
        });

    let denom = norm_a_final.sqrt() * norm_b_final.sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    1.0 - (dot_final / denom)
}

/// Optimized dot product for f32 vectors.
#[inline]
fn dot_product_f32_optimized(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    if len < UNROLL {
        return -a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>();
    }

    let unrolled_len = len - (len % UNROLL);

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in (0..unrolled_len).step_by(UNROLL) {
        sum0 += unsafe { *a_ptr.add(i) * *b_ptr.add(i) };
        sum1 += unsafe { *a_ptr.add(i + 1) * *b_ptr.add(i + 1) };
        sum2 += unsafe { *a_ptr.add(i + 2) * *b_ptr.add(i + 2) };
        sum3 += unsafe { *a_ptr.add(i + 3) * *b_ptr.add(i + 3) };
        sum0 += unsafe { *a_ptr.add(i + 4) * *b_ptr.add(i + 4) };
        sum1 += unsafe { *a_ptr.add(i + 5) * *b_ptr.add(i + 5) };
        sum2 += unsafe { *a_ptr.add(i + 6) * *b_ptr.add(i + 6) };
        sum3 += unsafe { *a_ptr.add(i + 7) * *b_ptr.add(i + 7) };
    }

    let sum = sum0 + sum1 + sum2 + sum3;

    // Handle remaining elements
    let remaining = a[unrolled_len..]
        .iter()
        .zip(b[unrolled_len..].iter())
        .map(|(&x, &y)| x * y)
        .sum::<f32>();

    -(sum + remaining)
}

/// Optimized Euclidean distance for f32 vectors.
#[inline]
fn euclidean_f32_optimized(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    if len < UNROLL {
        return a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum();
    }

    let unrolled_len = len - (len % UNROLL);

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in (0..unrolled_len).step_by(UNROLL) {
        let d0 = unsafe { *a_ptr.add(i) - *b_ptr.add(i) };
        sum0 += d0 * d0;

        let d1 = unsafe { *a_ptr.add(i + 1) - *b_ptr.add(i + 1) };
        sum1 += d1 * d1;

        let d2 = unsafe { *a_ptr.add(i + 2) - *b_ptr.add(i + 2) };
        sum2 += d2 * d2;

        let d3 = unsafe { *a_ptr.add(i + 3) - *b_ptr.add(i + 3) };
        sum3 += d3 * d3;

        let d4 = unsafe { *a_ptr.add(i + 4) - *b_ptr.add(i + 4) };
        sum0 += d4 * d4;

        let d5 = unsafe { *a_ptr.add(i + 5) - *b_ptr.add(i + 5) };
        sum1 += d5 * d5;

        let d6 = unsafe { *a_ptr.add(i + 6) - *b_ptr.add(i + 6) };
        sum2 += d6 * d6;

        let d7 = unsafe { *a_ptr.add(i + 7) - *b_ptr.add(i + 7) };
        sum3 += d7 * d7;
    }

    let sum = sum0 + sum1 + sum2 + sum3;

    // Handle remaining elements
    let remaining = a[unrolled_len..]
        .iter()
        .zip(b[unrolled_len..].iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f32>();

    sum + remaining
}

/// Cosine distance: 1 - cos(a, b)
/// Range: [0, 2] where 0 means identical direction, 2 means opposite.
pub struct Cosine;

impl Distance<f32> for Cosine {
    #[inline]
    fn compute(a: &[f32], b: &[f32]) -> f32 {
        cosine_f32_optimized(a, b)
    }
}

impl Distance<f64> for Cosine {
    fn compute(a: &[f64], b: &[f64]) -> f32 {
        let (dot, norm_a, norm_b) = a.iter()
            .zip(b.iter())
            .fold((0.0f64, 0.0f64, 0.0f64), |(d, na, nb), (&x, &y)| {
                (d + x * y, na + x * x, nb + y * y)
            });

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            return 0.0;
        }

        (1.0 - (dot / denom)) as f32
    }
}

impl Distance<i32> for Cosine {
    fn compute(a: &[i32], b: &[i32]) -> f32 {
        let (dot, norm_a, norm_b) = a.iter()
            .zip(b.iter())
            .fold((0.0f64, 0.0f64, 0.0f64), |(d, na, nb), (&x, &y)| {
                (d + (x as f64) * (y as f64), na + (x as f64) * (x as f64), nb + (y as f64) * (y as f64))
            });

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            return 0.0;
        }

        (1.0 - (dot / denom)) as f32
    }
}

impl Distance<i64> for Cosine {
    fn compute(a: &[i64], b: &[i64]) -> f32 {
        let (dot, norm_a, norm_b) = a.iter()
            .zip(b.iter())
            .fold((0.0f64, 0.0f64, 0.0f64), |(d, na, nb), (&x, &y)| {
                (d + (x as f64) * (y as f64), na + (x as f64) * (x as f64), nb + (y as f64) * (y as f64))
            });

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            return 0.0;
        }

        (1.0 - (dot / denom)) as f32
    }
}

/// Dot product distance: -dot(a, b)
/// Negative so that smaller values = more similar.
/// Use this when vectors are already normalized.
pub struct DotProduct;

impl Distance<f32> for DotProduct {
    #[inline]
    fn compute(a: &[f32], b: &[f32]) -> f32 {
        dot_product_f32_optimized(a, b)
    }
}

impl Distance<f64> for DotProduct {
    fn compute(a: &[f64], b: &[f64]) -> f32 {
        -(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f64>() as f32)
    }
}

impl Distance<i32> for DotProduct {
    fn compute(a: &[i32], b: &[i32]) -> f32 {
        -(a.iter().zip(b.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum::<f64>() as f32)
    }
}

impl Distance<i64> for DotProduct {
    fn compute(a: &[i64], b: &[i64]) -> f32 {
        -(a.iter().zip(b.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum::<f64>() as f32)
    }
}

/// Euclidean (L2) distance: sqrt(sum((a-b)^2))
/// Returns squared distance to avoid sqrt for comparisons.
pub struct Euclidean;

impl Distance<f32> for Euclidean {
    #[inline]
    fn compute(a: &[f32], b: &[f32]) -> f32 {
        euclidean_f32_optimized(a, b)
    }
}

impl Distance<f64> for Euclidean {
    fn compute(a: &[f64], b: &[f64]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>() as f32
    }
}

impl Distance<i32> for Euclidean {
    fn compute(a: &[i32], b: &[i32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| ((x - y) as f64).powi(2))
            .sum::<f64>() as f32
    }
}

impl Distance<i64> for Euclidean {
    fn compute(a: &[i64], b: &[i64]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| ((x - y) as f64).powi(2))
            .sum::<f64>() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_f32_identical() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        let dist = Cosine::compute(&a, &b);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_f32_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let dist = Cosine::compute(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_f32_opposite() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0];
        let dist = Cosine::compute(&a, &b);
        assert!((dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_f32_large() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32 / 128.0).collect();
        let dist = Cosine::compute(&a, &b);
        assert!(dist >= 0.0 && dist <= 2.0);
    }

    #[test]
    fn test_cosine_i32() {
        let a = vec![1i32, 0, 0];
        let b = vec![0i32, 1, 0];
        let dist = Cosine::compute(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 1.0, 1.0];
        let dist = DotProduct::compute(&a, &b);
        assert!((dist - (-6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_f32_large() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let dist = DotProduct::compute(&a, &b);
        assert!(dist < 0.0); // Should be negative
    }

    #[test]
    fn test_dot_product_i32() {
        let a = vec![1i32, 2, 3];
        let b = vec![1i32, 1, 1];
        let dist = DotProduct::compute(&a, &b);
        assert!((dist - (-6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_f32() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        let dist = Euclidean::compute(&a, &b);
        assert!((dist - 25.0).abs() < 1e-6); // 3^2 + 4^2 = 25
    }

    #[test]
    fn test_euclidean_f32_large() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();
        let dist = Euclidean::compute(&a, &b);
        // Each element differs by 1, so sum of 128 1s = 128
        assert!((dist - 128.0).abs() < 1e-4);
    }

    #[test]
    fn test_euclidean_i32() {
        let a = vec![0i32, 0];
        let b = vec![3i32, 4];
        let dist = Euclidean::compute(&a, &b);
        assert!((dist - 25.0).abs() < 1e-6);
    }
}

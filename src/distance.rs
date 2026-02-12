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

/// Cosine distance: 1 - cos(a, b)
/// Range: [0, 2] where 0 means identical direction, 2 means opposite.
pub struct Cosine;

impl<T: Numeric> Distance<T> for Cosine {
    fn compute(a: &[T], b: &[T]) -> f32 {
        let (dot, norm_a, norm_b) = a.iter()
            .zip(b.iter())
            .fold((0.0f32, 0.0f32, 0.0f32), |(d, na, nb), (&x, &y)| {
                let xf = x.to_f32();
                let yf = y.to_f32();
                (d + xf * yf, na + xf * xf, nb + yf * yf)
            });

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            return 0.0;
        }

        1.0 - (dot / denom)
    }
}

/// Dot product distance: -dot(a, b)
/// Negative so that smaller values = more similar.
/// Use this when vectors are already normalized.
pub struct DotProduct;

impl<T: Numeric> Distance<T> for DotProduct {
    fn compute(a: &[T], b: &[T]) -> f32 {
        -a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x.to_f32() * y.to_f32())
            .sum::<f32>()
    }
}

/// Euclidean (L2) distance: sqrt(sum((a-b)^2))
/// Returns squared distance to avoid sqrt for comparisons.
pub struct Euclidean;

impl<T: Numeric> Distance<T> for Euclidean {
    fn compute(a: &[T], b: &[T]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x.to_f32() - y.to_f32()).powi(2))
            .sum()
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
    fn test_cosine_i32() {
        let a = vec![1i32, 0, 0];
        let b = vec![0i32, 1, 0];
        let dist = Cosine::compute(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_i32() {
        let a = vec![1i32, 2, 3];
        let b = vec![1i32, 1, 1];
        let dist = DotProduct::compute(&a, &b);
        assert!((dist - (-6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_i32() {
        let a = vec![0i32, 0];
        let b = vec![3i32, 4];
        let dist = Euclidean::compute(&a, &b);
        assert!((dist - 25.0).abs() < 1e-6);
    }
}

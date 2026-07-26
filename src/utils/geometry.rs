//! # 📐 Geometry & Projection Utilities
//!
//! Inverse Perspective Mapping (IPM), pinhole distance estimation,
//! and bounding-box helpers for the driving vision pipeline.
//!
//! ## TODO
//!
//! - [ ] Implement pinhole distance: `Z = (f * W) / w`
//! - [ ] Add IPM for BEV grid projection
//! - [ ] Add IoU computation for evaluation
//! - [ ] Write property-based tests (e.g., "larger bbox → closer")

#![allow(unused_variables, dead_code)]

/// Estimates distance to an object using the pinhole camera model.
///
/// # Arguments
/// * `pixel_width` - Width of the object's bounding box in pixels.
/// * `real_width` - Known real-world width (e.g., car ≈ 1.8 m).
/// * `focal_length` - Camera focal length in pixels.
///
/// # Returns
/// Distance Z in meters.
pub fn estimate_distance(pixel_width: f32, real_width: f32, focal_length: f32) -> f32 {
    todo!("Z = (focal_length * real_width) / pixel_width");
}

/// Computes relative velocity from distance change over time.
///
/// # Returns
/// Relative velocity (m/s). Positive = approaching, negative = receding.
pub fn compute_relative_velocity(prev_distance: f32, curr_distance: f32, dt: f32) -> f32 {
    todo!("V_rel = (curr_distance - prev_distance) / dt");
}

/// First-order low-pass filter for smoothing noisy measurements.
///
/// * `alpha` in [0, 1] — 0 = max smooth, 1 = no smoothing.
pub fn low_pass_filter(value: f32, prev_value: f32, alpha: f32) -> f32 {
    alpha * value + (1.0 - alpha) * prev_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "not implemented")]
    fn test_estimate_distance_stub() {
        let _ = estimate_distance(100.0, 1.8, 650.0);
    }

    #[test]
    fn test_low_pass_filter_no_smoothing() {
        let result = low_pass_filter(10.0, 5.0, 1.0);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_low_pass_filter_full_smoothing() {
        let result = low_pass_filter(10.0, 5.0, 0.0);
        assert!((result - 5.0).abs() < 1e-6);
    }
}

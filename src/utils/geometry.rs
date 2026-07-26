//! # 📐 Geometry & Projection Utilities
//!
//! Inverse Perspective Mapping (IPM), pinhole distance estimation,
//! and bounding-box helpers for the driving vision pipeline.

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
    if pixel_width <= 0.0 {
        return f32::MAX;
    }
    (focal_length * real_width) / pixel_width
}

/// Computes relative velocity from distance change over time.
///
/// # Returns
/// Relative velocity (m/s). Positive = approaching, negative = receding.
pub fn compute_relative_velocity(prev_distance: f32, curr_distance: f32, dt: f32) -> f32 {
    if dt <= 0.0 {
        return 0.0;
    }
    (prev_distance - curr_distance) / dt
}

/// First-order low-pass filter for smoothing noisy measurements.
///
/// * `alpha` in [0, 1] — 0 = max smooth, 1 = no smoothing.
pub fn low_pass_filter(value: f32, prev_value: f32, alpha: f32) -> f32 {
    alpha * value + (1.0 - alpha) * prev_value
}

/// Computes Intersection-over-Union between two bounding boxes.
pub fn compute_iou(
    a: (f32, f32, f32, f32),
    b: (f32, f32, f32, f32),
) -> f32 {
    let (ax1, ay1, ax2, ay2) = a;
    let (bx1, by1, bx2, by2) = b;

    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);

    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;

    let a_area = (ax2 - ax1) * (ay2 - ay1);
    let b_area = (bx2 - bx1) * (by2 - by1);
    let union = a_area + b_area - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Converts a detection's bounding box to (center_x, center_y, width, height) format.
pub fn bbox_to_cxcywh(x1: f32, y1: f32, x2: f32, y2: f32) -> (f32, f32, f32, f32) {
    let cx = (x1 + x2) / 2.0;
    let cy = (y1 + y2) / 2.0;
    let w = (x2 - x1).abs();
    let h = (y2 - y1).abs();
    (cx, cy, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_distance_basic() {
        // A car of 1.8m width occupying 180 pixels with focal length 650.
        let dist = estimate_distance(180.0, 1.8, 650.0);
        // Z = (650 * 1.8) / 180 = 6.5m
        assert!((dist - 6.5).abs() < 1e-4);
    }

    #[test]
    fn test_estimate_distance_larger_bbox_closer() {
        // A larger bbox → smaller distance.
        let far = estimate_distance(50.0, 1.8, 650.0);
        let close = estimate_distance(500.0, 1.8, 650.0);
        assert!(close < far);
    }

    #[test]
    fn test_compute_relative_velocity_approaching() {
        // Object moved from 20m to 10m in 1 second → approaching at 10 m/s.
        let vel = compute_relative_velocity(20.0, 10.0, 1.0);
        assert!((vel - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_relative_velocity_receding() {
        // Object moved from 10m to 20m in 1 second → receding at -10 m/s.
        let vel = compute_relative_velocity(10.0, 20.0, 1.0);
        assert!((vel - (-10.0)).abs() < 1e-4);
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

    #[test]
    fn test_compute_iou_same_box() {
        let iou = compute_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0));
        assert!((iou - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_iou_no_overlap() {
        let iou = compute_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0));
        assert!((iou - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_iou_half_overlap() {
        let iou = compute_iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0));
        // Overlap: 5×10 = 50. Union: 100+100-50 = 150. IoU = 50/150 = 0.333
        assert!((iou - 50.0 / 150.0).abs() < 1e-4);
    }
}

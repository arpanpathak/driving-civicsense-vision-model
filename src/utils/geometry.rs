//! # 📐 Geometry & Projection Utilities
//!
//! Shared math helpers used across the detection, tracking, and analysis
//! modules:
//!
//! - Pinhole camera distance estimation
//! - Relative velocity from frame-to-frame distance change
//! - First-order low-pass filter for smoothing
//! - Intersection-over-Union (IoU) between bounding boxes
//! - Bounding-box format conversion

// ─────────────────────────────────────────────────────────────────────────────
//  Distance estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Estimates the distance to an object using the pinhole camera model.
///
/// **Formula:** `Z = (focal_length × real_width) / pixel_width`
///
/// # Parameters
/// - `pixel_width`, Width of the object's bounding box in **pixels**.
///   Must be > 0; returns `f32::MAX` if ≤ 0.
/// - `real_width`, Known real-world width of the object in **meters**
///   (e.g., car ≈ 1.8 m, stop sign ≈ 0.75 m).
/// - `focal_length`, Camera focal length in **pixels** (from calibration).
///
/// # Returns
/// Estimated distance **Z** in meters.  Larger values mean the object is
/// farther away.  Returns `f32::MAX` for degenerate (zero/negative) pixel
/// widths.
///
/// # Example
/// ```
/// use civicsense::utils::geometry::estimate_distance;
/// // A car 1.8 m wide spanning 180 px on a 650 px focal-length camera:
/// let dist = estimate_distance(180.0, 1.8, 650.0);
/// assert!((dist - 6.5).abs() < 1e-4); // 6.5 m
/// ```
pub fn estimate_distance(pixel_width: f32, real_width: f32, focal_length: f32) -> f32 {
    if pixel_width <= 0.0 {
        return f32::MAX;
    }
    (focal_length * real_width) / pixel_width
}

// ─────────────────────────────────────────────────────────────────────────────
//  Relative velocity
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the relative velocity of an object from its distance change
/// between two consecutive frames.
///
/// **Formula:** `V_rel = (prev_distance - curr_distance) / dt`
///
/// # Parameters
/// - `prev_distance`, Object distance in the **previous** frame (meters).
/// - `curr_distance`, Object distance in the **current** frame (meters).
/// - `dt`, Time elapsed between the two frames (**seconds**). Must be > 0;
///   returns `0.0` if ≤ 0.
///
/// # Returns
/// Relative velocity in **meters per second**.
///
/// | Sign  | Meaning                      |
/// |-------|------------------------------|
/// | +     | Object is **approaching**    |
/// | -     | Object is **receding**       |
/// | 0     | No change (or dt = 0)        |
///
/// # Example
/// ```
/// use civicsense::utils::geometry::compute_relative_velocity;
/// // Object moved from 20 m to 10 m in 1 second:
/// let vel = compute_relative_velocity(20.0, 10.0, 1.0);
/// assert!((vel - 10.0).abs() < 1e-4); // 10 m/s approaching
/// ```
pub fn compute_relative_velocity(prev_distance: f32, curr_distance: f32, dt: f32) -> f32 {
    if dt <= 0.0 {
        return 0.0;
    }
    (prev_distance - curr_distance) / dt
}

// ─────────────────────────────────────────────────────────────────────────────
//  Low-pass filter
// ─────────────────────────────────────────────────────────────────────────────

/// First-order infinite-impulse-response (IIR) low-pass filter.
///
/// **Formula:** `output = alpha × value + (1 - alpha) × prev_value`
///
/// # Parameters
/// - `value`, New (raw) measurement.
/// - `prev_value`, Filtered output from the previous time step.
/// - `alpha`, Smoothing factor in `[0, 1]`:
///   - `alpha = 1.0` → no smoothing (pass-through).
///   - `alpha = 0.0` → output sticks to `prev_value` forever.
///   - Typical value: `0.3 – 0.5`.
///
/// # Returns
/// The filtered value.
///
/// # Example
/// ```
/// use civicsense::utils::geometry::low_pass_filter;
/// let smooth = low_pass_filter(10.0, 5.0, 0.3);
/// assert!((smooth - 6.5).abs() < 1e-6);
/// ```
pub fn low_pass_filter(value: f32, prev_value: f32, alpha: f32) -> f32 {
    alpha * value + (1.0 - alpha) * prev_value
}

// ─────────────────────────────────────────────────────────────────────────────
//  IoU
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the Intersection-over-Union of two axis-aligned bounding boxes.
///
/// # Parameters
/// - `a`, First box `(x1, y1, x2, y2)` in pixel coordinates.
/// - `b`, Second box `(x1, y1, x2, y2)` in pixel coordinates.
///
/// # Returns
/// IoU in `[0.0, 1.0]`:
/// - `1.0` → identical boxes.
/// - `0.0` → no overlap.
///
/// # Example
/// ```
/// use civicsense::utils::geometry::compute_iou;
/// let iou = compute_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0));
/// assert!((iou - 1.0).abs() < 1e-6);
/// ```
pub fn compute_iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
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

    if union <= 0.0 { 0.0 } else { inter / union }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bbox conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Converts a bounding box from `(x1, y1, x2, y2)` to
/// `(center_x, center_y, width, height)` format.
///
/// This format is used internally by the Kalman filter state vector.
///
/// # Parameters
/// - `x1`, Left edge (pixels).
/// - `y1`, Top edge (pixels).
/// - `x2`, Right edge (pixels).
/// - `y2`, Bottom edge (pixels).
///
/// # Returns
/// A tuple `(cx, cy, w, h)`.
///
/// # Example
/// ```
/// use civicsense::utils::geometry::bbox_to_cxcywh;
/// let (cx, cy, w, h) = bbox_to_cxcywh(0.0, 0.0, 100.0, 200.0);
/// assert_eq!((cx, cy, w, h), (50.0, 100.0, 100.0, 200.0));
/// ```
pub fn bbox_to_cxcywh(x1: f32, y1: f32, x2: f32, y2: f32) -> (f32, f32, f32, f32) {
    let cx = (x1 + x2) / 2.0;
    let cy = (y1 + y2) / 2.0;
    let w = (x2 - x1).abs();
    let h = (y2 - y1).abs();
    (cx, cy, w, h)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_distance_basic() {
        // A car of 1.8 m width occupying 180 pixels with focal length 650.
        let dist = estimate_distance(180.0, 1.8, 650.0);
        // Z = (650 * 1.8) / 180 = 6.5 m
        assert!((dist - 6.5).abs() < 1e-4);
    }

    #[test]
    fn test_estimate_distance_larger_bbox_closer() {
        let far = estimate_distance(50.0, 1.8, 650.0);
        let close = estimate_distance(500.0, 1.8, 650.0);
        assert!(close < far);
    }

    #[test]
    fn test_compute_relative_velocity_approaching() {
        let vel = compute_relative_velocity(20.0, 10.0, 1.0);
        assert!((vel - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_relative_velocity_receding() {
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
        assert!((iou - 50.0 / 150.0).abs() < 1e-4);
    }
}

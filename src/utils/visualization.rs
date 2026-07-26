//! # 🎨 Visualization Utilities
//!
//! Debug overlay rendering for bounding boxes, alert banners, and status
//! indicators.  Used during development and validation (`--visualize` flag).
//!
//! ## Implementation Note
//!
//! Because we avoid pulling in a full graphics library (OpenCV, etc.),
//! text rendering is **not available**.  Instead:
//!
//! - Bounding boxes are drawn as coloured rectangles with a filled "label
//!   bar" at the top.
//! - Alerts are shown as a coloured strip across the top 20 rows of the
//!   frame plus a log message.

use crate::detection::yolo::Detection;

// ─────────────────────────────────────────────────────────────────────────────
//  Colour palette
// ─────────────────────────────────────────────────────────────────────────────

/// Red — used for stop signs (class 0).
const STOP_SIGN_COLOR: (u8, u8, u8) = (255, 0, 0);

/// Yellow — used for traffic lights and crosswalks (classes 1, 2).
const TRAFFIC_LIGHT_COLOR: (u8, u8, u8) = (255, 255, 0);

/// Green — used for vehicles (classes 3, 4, 5).
const VEHICLE_COLOR: (u8, u8, u8) = (0, 255, 0);

/// Gray — fallback for unknown classes.
const DEFAULT_COLOR: (u8, u8, u8) = (128, 128, 128);

/// Returns the display colour for a given class ID.
///
/// # Parameters
/// - `class_id` — YOLO class index (0 = stop_sign, 1-2 = traffic stuff,
///   3-5 = vehicles, etc.).
///
/// # Returns
/// An `(R, G, B)` tuple.
fn class_color(class_id: u32) -> (u8, u8, u8) {
    match class_id {
        0 => STOP_SIGN_COLOR,
        1 | 2 => TRAFFIC_LIGHT_COLOR,
        3 | 4 | 5 => VEHICLE_COLOR,
        _ => DEFAULT_COLOR,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pixel-level drawing helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Sets a single pixel in a flattened RGB8 buffer.
///
/// # Parameters
/// - `frame` — Mutable RGB8 pixel buffer `(H × W × 3)`.
/// - `x` — Column (0-based, clamped to `[0, width)`).
/// - `y` — Row (0-based, clamped to `[0, height)`).
/// - `width` — Frame width in pixels.
/// - `height` — Frame height in pixels.
/// - `color` — `(R, G, B)` tuple.  Values are written as-is (no clamping).
///
/// # Safety
/// Bounds-checked: silently returns if `(x, y)` is out of range or the
/// buffer is too small.
fn set_pixel(frame: &mut [u8], x: u32, y: u32, width: u32, height: u32, color: (u8, u8, u8)) {
    if x >= width || y >= height {
        return;
    }
    let idx = ((y * width + x) * 3) as usize;
    if idx + 2 < frame.len() {
        frame[idx] = color.0;
        frame[idx + 1] = color.1;
        frame[idx + 2] = color.2;
    }
}

/// Draws a rectangular outline on the frame buffer.
///
/// # Parameters
/// - `frame` — Mutable RGB8 buffer.
/// - `x1`, `y1` — Top-left corner (clamped to frame bounds).
/// - `x2`, `y2` — Bottom-right corner (clamped to frame bounds).
/// - `width`, `height` — Frame dimensions.
/// - `color` — `(R, G, B)` colour.
/// - `thickness` — Line thickness in pixels (≥ 1).
fn draw_rect(
    frame: &mut [u8],
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
    width: u32,
    height: u32,
    color: (u8, u8, u8),
    thickness: u32,
) {
    for t in 0..thickness {
        // Top and bottom horizontal lines.
        for x in x1..=x2 {
            set_pixel(frame, x, y1 + t, width, height, color);
            set_pixel(frame, x, y2.saturating_sub(t), width, height, color);
        }
        // Left and right vertical lines.
        for y in y1..=y2 {
            set_pixel(frame, x1 + t, y, width, height, color);
            set_pixel(frame, x2.saturating_sub(t), y, width, height, color);
        }
    }
}

/// Fills a horizontal strip of pixels with a solid colour.
///
/// Used to draw the alert banner at the top of the frame.
///
/// # Parameters
/// - `frame` — Mutable RGB8 buffer.
/// - `y_start` — First row to fill (inclusive).
/// - `y_end` — Last row to fill (exclusive, clamped to `height`).
/// - `width`, `height` — Frame dimensions.
/// - `color` — `(R, G, B)` fill colour.
fn fill_strip(
    frame: &mut [u8],
    y_start: u32,
    y_end: u32,
    width: u32,
    height: u32,
    color: (u8, u8, u8),
) {
    for y in y_start..y_end.min(height) {
        for x in 0..width {
            set_pixel(frame, x, y, width, height, color);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Draws detection bounding boxes and class-colour bars on a raw frame.
///
/// Each detection is rendered as:
/// - A **rectangular outline** (2 px thick) colour-coded by class.
/// - A **filled label bar** (12 px tall) at the top of the bounding box,
///   spanning up to 80 px wide.
///
/// # Parameters
/// - `frame` — Mutable RGB8 frame buffer `(H × W × 3)`.  Modified in place.
/// - `width` — Frame width in pixels.
/// - `height` — Frame height in pixels.
/// - `detections` — Slice of [`Detection`] objects to render.
/// - `class_names` — Class-name lookup table (index → display name).
///   Only used for debug logging; text is not drawn on the buffer.
///
/// # Panics
/// Never panics.  Out-of-bounds coordinates are silently clamped.
pub fn draw_detections(
    frame: &mut [u8],
    width: u32,
    height: u32,
    detections: &[Detection],
    class_names: &[String],
) {
    for det in detections {
        let x1 = (det.x1 as u32).clamp(0, width.saturating_sub(1));
        let y1 = (det.y1 as u32).clamp(0, height.saturating_sub(1));
        let x2 = (det.x2 as u32).clamp(x1 + 1, width.saturating_sub(1));
        let y2 = (det.y2 as u32).clamp(y1 + 1, height.saturating_sub(1));

        let color = class_color(det.class_id);
        draw_rect(frame, x1, y1, x2, y2, width, height, color, 2);

        let label = class_names
            .get(det.class_id as usize)
            .cloned()
            .unwrap_or_else(|| format!("cls_{}", det.class_id));

        // Filled label bar at the top of the bounding box.
        let bar_h = 12u32;
        let bar_y2 = (y1 + bar_h).min(y2);
        for row in y1..bar_y2 {
            for col in x1..(x1 + 80).min(x2) {
                set_pixel(frame, col, row, width, height, color);
            }
        }

        log::debug!("  Detection: {} at ({}, {}, {}, {})", label, x1, y1, x2, y2);
    }
}

/// Draws a coloured alert banner at the top of the frame.
///
/// Because raw-pixel buffers lack text rendering, the alert is shown as a
/// solid-colour strip and also logged via `log::info!`.
///
/// # Colour coding
///
/// | Alert text contains | Colour       | Meaning   |
/// |---------------------|--------------|-----------|
/// | `"STOP"` / `"BLOCKED"` | Red        | Critical  |
/// | `"MERGE"`           | Orange       | Courtesy  |
/// | (anything else)     | Yellow       | Default   |
///
/// # Parameters
/// - `frame` — Mutable RGB8 buffer.  The top 20 rows are overwritten.
/// - `width` — Frame width in pixels.
/// - `height` — Frame height in pixels.
/// - `text` — Alert description.  Used for colour selection and logging.
///
/// # Panics
/// Never panics.
pub fn draw_alert_text(frame: &mut [u8], width: u32, height: u32, text: &str) {
    let alert_color = match text {
        t if t.contains("STOP") || t.contains("BLOCKED") => (255, 0, 0),
        t if t.contains("MERGE") => (255, 165, 0),
        _ => (255, 255, 0),
    };

    fill_strip(frame, 0, 20, width, height, alert_color);
    log::info!("🚦 ALERT: {}", text);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Drawing detections on a blank frame should modify some pixels.
    #[test]
    fn test_draw_detections_no_panic() {
        let mut frame = vec![0u8; 640 * 480 * 3];
        let dets = vec![Detection {
            x1: 10.0,
            y1: 10.0,
            x2: 100.0,
            y2: 100.0,
            confidence: 0.9,
            class_id: 0,
        }];
        let names = vec!["stop_sign".to_string()];
        draw_detections(&mut frame, 640, 480, &dets, &names);
        let has_color = frame.iter().any(|&p| p != 0);
        assert!(has_color);
    }

    /// Drawing alert text on a blank frame should modify some pixels.
    #[test]
    fn test_draw_alert_text_no_panic() {
        let mut frame = vec![0u8; 640 * 480 * 3];
        draw_alert_text(&mut frame, 640, 480, "STOP SIGN VIOLATION");
        let has_color = frame.iter().any(|&p| p != 0);
        assert!(has_color);
    }
}

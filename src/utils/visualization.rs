//! # 🎨 Visualization Utilities
//!
//! Debug overlay rendering for bounding boxes, track IDs, speed labels,
//! and the BEV occupancy grid. Used during development and validation.
//!
//! ## Implementation Note
//!
//! Raw pixel buffer drawing (without a graphics library) supports only
//! solid-color overlays — text rendering requires `image` crate font
//! support or OpenCV. For development, we draw colored rectangles to
//! indicate detections and alerts.

use crate::detection::yolo::Detection;

/// Colors for different classes (RGB tuples).
const STOP_SIGN_COLOR: (u8, u8, u8) = (255, 0, 0);    // Red
const TRAFFIC_LIGHT_COLOR: (u8, u8, u8) = (255, 255, 0); // Yellow
const VEHICLE_COLOR: (u8, u8, u8) = (0, 255, 0);       // Green
const DEFAULT_COLOR: (u8, u8, u8) = (128, 128, 128);   // Gray

fn class_color(class_id: u32) -> (u8, u8, u8) {
    match class_id {
        0 => STOP_SIGN_COLOR,
        1 | 2 => TRAFFIC_LIGHT_COLOR,
        3 | 4 | 5 => VEHICLE_COLOR,
        _ => DEFAULT_COLOR,
    }
}

/// Sets a pixel in the frame buffer at (x, y) to the given RGB color.
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

/// Draws a rectangle outline on the frame buffer.
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

/// Fills a horizontal strip of pixels with a solid color (for alert banners).
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

/// Draws detection bounding boxes and labels on a raw frame buffer.
///
/// # Arguments
/// * `frame` - Mutable RGB8 frame buffer (flattened, H×W×3).
/// * `width` - Image width in pixels.
/// * `height` - Image height in pixels.
/// * `detections` - Detections to render.
/// * `class_names` - Mapping of class_id → display name.
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

        // Draw a small filled rectangle at the top of the bbox for a "label bar".
        let label = class_names
            .get(det.class_id as usize)
            .cloned()
            .unwrap_or_else(|| format!("cls_{}", det.class_id));

        // Just color the top-left corner area to indicate the class.
        let bar_h = 12u32;
        let bar_y2 = (y1 + bar_h).min(y2);
        for row in y1..bar_y2 {
            for col in x1..(x1 + 80).min(x2) {
                set_pixel(frame, col, row, width, height, color);
                // Slightly dim the bar so text (if drawn) could be visible.
            }
        }

        // TODO: Render actual text. Requires a font rasterizer or OpenCV.
        log::debug!("  Detection: {} at ({}, {}, {}, {})", label, x1, y1, x2, y2);
    }
}

/// Draws alert text in the top-left corner of the frame.
///
/// Since we don't have text rendering in raw pixel buffers, we draw a
/// colored banner at the top of the frame to indicate alert state.
pub fn draw_alert_text(frame: &mut [u8], width: u32, height: u32, text: &str) {
    let alert_color = match text {
        t if t.contains("STOP") || t.contains("BLOCKED") => (255, 0, 0),   // Red for critical
        t if t.contains("MERGE") => (255, 165, 0),                          // Orange for courtesy
        _ => (255, 255, 0),                                                 // Yellow default
    };

    // Fill a banner across the top 20 pixels.
    fill_strip(frame, 0, 20, width, height, alert_color);

    // Log the alert so the user can see it in the console.
    log::info!("🚦 ALERT: {}", text);
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Should not panic.
        draw_detections(&mut frame, 640, 480, &dets, &names);
        // Some pixels should have been modified.
        let has_color = frame.iter().any(|&p| p != 0);
        assert!(has_color);
    }

    #[test]
    fn test_draw_alert_text_no_panic() {
        let mut frame = vec![0u8; 640 * 480 * 3];
        draw_alert_text(&mut frame, 640, 480, "STOP SIGN VIOLATION");
        let has_color = frame.iter().any(|&p| p != 0);
        assert!(has_color);
    }
}

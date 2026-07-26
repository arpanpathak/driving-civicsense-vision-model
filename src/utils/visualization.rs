//! # 🎨 Visualization Utilities
//!
//! Debug overlay rendering for bounding boxes, track IDs, speed labels,
//! and the BEV occupancy grid. Used during development and validation.
//!
//! ## TODO
//!
//! - [ ] Draw bounding boxes with class labels and confidence
//! - [ ] Render track IDs with speed overlays
//! - [ ] Draw BEV occupancy grid
//! - [ ] Color-code alerts (green/yellow/red)

#![allow(unused_variables, dead_code)]

use crate::detection::yolo::Detection;

/// Draws detection bounding boxes and labels on a raw frame buffer.
///
/// # Arguments
/// * `frame` - Mutable RGB8 frame buffer (flattened, H×W×3).
/// * `width` - Image width.
/// * `height` - Image height.
/// * `detections` - Detections to render.
/// * `class_names` - Mapping of class_id → display name.
pub fn draw_detections(
    frame: &mut [u8],
    width: u32,
    height: u32,
    detections: &[Detection],
    class_names: &[String],
) {
    todo!("Implement bounding box + label overlay");
}

/// Draws alert text in the top-left corner of the frame.
pub fn draw_alert_text(frame: &mut [u8], width: u32, height: u32, text: &str) {
    todo!("Implement alert text overlay");
}

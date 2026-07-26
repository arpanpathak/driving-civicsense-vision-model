//! YOLOv8 / YOLOv11 ONNX object detector.
//!
//! Loads an ONNX model via `ort`, pre-processes frames (letterbox, normalize,
//! HWC→CHW), runs inference, and post-processes (grid decode, sigmoid,
//! confidence filter, NMS, scale to original dimensions).
//!
//! ## Graceful degradation
//!
//! If no model file exists at the configured path, the detector constructs
//! successfully and returns empty results.  This allows pipeline development
//! and data collection before a custom model is trained.

use std::path::Path;

use crate::config::ModelConfig;

// ─────────────────────────────────────────────────────────────────────────────
//  Detection
// ─────────────────────────────────────────────────────────────────────────────

/// A single object detection produced by the YOLO model.
///
/// Coordinates are **absolute pixel values** in the original (un-resized)
/// image.  Box convention: `(x1, y1)` = top-left, `(x2, y2)` = bottom-right.
#[derive(Debug, Clone)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
//  YoloConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration parameters for the YOLO detector.
#[derive(Debug, Clone)]
pub struct YoloConfig {
    pub model_path: String,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    pub input_width: u32,
    pub input_height: u32,
    pub class_names: Vec<String>,
}

impl From<&ModelConfig> for YoloConfig {
    fn from(cfg: &ModelConfig) -> Self {
        Self {
            model_path: cfg.path.clone(),
            conf_threshold: cfg.conf_threshold,
            iou_threshold: cfg.iou_threshold,
            input_width: cfg.input_width,
            input_height: cfg.input_height,
            class_names: cfg.classes.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pre-processing
// ─────────────────────────────────────────────────────────────────────────────

/// Result of letterbox pre-processing.
struct LetterBox {
    /// Float32 CHW tensor normalized to [0, 1], shape [3, H, W].
    tensor: Vec<f32>,
    /// Scale factor from original → model input.
    scale: f32,
    /// Horizontal padding applied (pixels in model-input space).
    pad_x: f32,
    /// Vertical padding applied (pixels in model-input space).
    pad_y: f32,
}

/// Letterbox resize: scale to fit within `dst_w × dst_h` while maintaining
/// aspect ratio, then pad to exactly `dst_w × dst_h` with gray (114).
///
/// Returns the CHW float32 tensor and metadata needed to map detections back
/// to the original image.
fn letterbox(
    frame: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> LetterBox {
    let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as u32;
    let new_h = (src_h as f32 * scale).round() as u32;
    let pad_x = (dst_w - new_w) as f32 / 2.0;
    let pad_y = (dst_h - new_h) as f32 / 2.0;

    // Build the RGB image from raw buffer for cropping/resizing.
    let src_img =
        image::RgbImage::from_raw(src_w, src_h, frame.to_vec()).expect("valid frame buffer");
    let resized = image::imageops::resize(
        &src_img,
        new_w,
        new_h,
        image::imageops::FilterType::CatmullRom,
    );

    // CHW float32 output, initialized to gray (114 / 255 ≈ 0.447).
    let mut tensor = vec![114.0f32 / 255.0; (dst_w * dst_h * 3) as usize];

    // Copy resized pixels into the center of the padded canvas.
    // tensor layout: CHW — channel 0 = R, 1 = G, 2 = B, each row-major.
    let total = (dst_w * dst_h) as usize;
    for y in 0..new_h {
        for x in 0..new_w {
            let pixel = resized.get_pixel(x, y);
            let canvas_x = (x as f32 + pad_x) as u32;
            let canvas_y = (y as f32 + pad_y) as u32;
            let idx = (canvas_y * dst_w + canvas_x) as usize;
            tensor[idx] = pixel[0] as f32 / 255.0;            // R
            tensor[total + idx] = pixel[1] as f32 / 255.0;    // G
            tensor[2 * total + idx] = pixel[2] as f32 / 255.0; // B
        }
    }

    LetterBox { tensor, scale, pad_x, pad_y }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Post-processing helpers
// ─────────────────────────────────────────────────────────────────────────────

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute IoU between two xyxy boxes.
fn box_iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let ix1 = a.0.max(b.0);
    let iy1 = a.1.max(b.1);
    let ix2 = a.2.min(b.2);
    let iy2 = a.3.min(b.3);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let area_a = (a.2 - a.0) * (a.3 - a.1);
    let area_b = (b.2 - b.0) * (b.3 - b.1);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Non-maximum suppression: greedily select highest-confidence boxes above
/// the IoU threshold.
fn non_max_suppression(
    mut boxes: Vec<(f32, f32, f32, f32, f32, u32)>,
    iou_threshold: f32,
) -> Vec<(f32, f32, f32, f32, f32, u32)> {
    // Sort descending by confidence.
    boxes.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();
    while !boxes.is_empty() {
        let best = boxes.remove(0);
        keep.push(best);
        boxes.retain(|b| box_iou((best.0, best.1, best.2, best.3), (b.0, b.1, b.2, b.3)) <= iou_threshold);
    }
    keep
}

// ─────────────────────────────────────────────────────────────────────────────
//  Grid / stride helpers for YOLOv8 decoding
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-computed anchor information for YOLOv8 decoding.
struct AnchorGrid {
    /// For each of the N predictions: (grid_x, grid_y, stride).
    anchors: Vec<(f32, f32, f32)>,
    /// Total number of predictions.
    num_predictions: usize,
}

impl AnchorGrid {
    /// Build the anchor grid for a given model input size.
    ///
    /// YOLOv8 produces predictions at 3 strides (8, 16, 32). The total
    /// number of predictions for a 640×640 input is 8400.
    fn new(input_size: u32) -> Self {
        let strides: [u32; 3] = [8, 16, 32];
        let mut anchors = Vec::new();

        for &stride in &strides {
            let grid_w = input_size / stride;
            let grid_h = input_size / stride;
            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    anchors.push((gx as f32, gy as f32, stride as f32));
                }
            }
        }

        let num_predictions = anchors.len();
        Self { anchors, num_predictions }
    }

    /// Decode raw YOLOv8 output into candidate detections.
    ///
    /// `output` is the raw float32 tensor from the ONNX model, shape
    /// `[1, 4 + num_classes, num_predictions]` in CHW layout.
    fn decode(
        &self,
        output: &[f32],
        num_classes: usize,
        conf_threshold: f32,
        orig_w: u32,
        orig_h: u32,
        scale: f32,
        pad_x: f32,
        pad_y: f32,
    ) -> Vec<(f32, f32, f32, f32, f32, u32)> {
        let stride_size = self.num_predictions;
        let mut candidates = Vec::new();

        for (i, &(grid_x, grid_y, stride)) in self.anchors.iter().enumerate() {
            // Read bbox: cx, cy, w, h at channel offsets 0..3.
            let cx_raw = output[i];
            let cy_raw = output[1 * stride_size + i];
            let w_raw = output[2 * stride_size + i];
            let h_raw = output[3 * stride_size + i];

            // YOLOv8 decoding:
            let cx = (sigmoid(cx_raw) * 2.0 - 0.5 + grid_x) * stride;
            let cy = (sigmoid(cy_raw) * 2.0 - 0.5 + grid_y) * stride;
            let w = (sigmoid(w_raw) * 2.0).powi(2) * stride;
            let h = (sigmoid(h_raw) * 2.0).powi(2) * stride;

            // Find best class.
            let mut best_conf = 0.0f32;
            let mut best_class = 0u32;
            for c in 0..num_classes {
                let score = sigmoid(output[(4 + c) * stride_size + i]);
                if score > best_conf {
                    best_conf = score;
                    best_class = c as u32;
                }
            }

            if best_conf < conf_threshold {
                continue;
            }

            // Convert cxcywh → xyxy in model-input coordinates.
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            // Remove padding and scale back to original image.
            let x1_orig = ((x1 - pad_x) / scale).max(0.0).min(orig_w as f32);
            let y1_orig = ((y1 - pad_y) / scale).max(0.0).min(orig_h as f32);
            let x2_orig = ((x2 - pad_x) / scale).max(0.0).min(orig_w as f32);
            let y2_orig = ((y2 - pad_y) / scale).max(0.0).min(orig_h as f32);

            if (x2_orig - x1_orig) < 1.0 || (y2_orig - y1_orig) < 1.0 {
                continue;
            }

            candidates.push((x1_orig, y1_orig, x2_orig, y2_orig, best_conf, best_class));
        }

        candidates
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  YoloDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Real-time YOLO object detector wrapping an ONNX Runtime session.
///
/// If the model file is absent, `detect()` returns empty results (graceful
/// degradation) so the pipeline can run for data collection or testing.
pub struct YoloDetector {
    config: YoloConfig,
    session: Option<ort::session::Session>,
    anchor_grid: AnchorGrid,
}

impl YoloDetector {
    /// Construct a new detector.
    ///
    /// Loads the ONNX model if it exists. A missing model is not an error —
    /// the detector simply returns empty detections.
    pub fn new(config: YoloConfig) -> Result<Self, String> {
        let model_path = Path::new(&config.model_path);

        let session = if model_path.exists() {
            log::info!("Loading ONNX model from '{}'", config.model_path);
            let session = ort::session::Session::builder()
                .map_err(|e| format!("ort init: {e}"))?
                .commit_from_file(model_path)
                .map_err(|e| format!("Failed to load model '{}': {e}", config.model_path))?;
            log::info!("Model loaded: {}x{} input", config.input_width, config.input_height);
            Some(session)
        } else {
            log::warn!(
                "ONNX model not found at '{}'. Detector returns empty results. \
                 Train a model (see CLOUD_TRAINING.md) and place it at this path.",
                config.model_path
            );
            None
        };

        let anchor_grid = AnchorGrid::new(config.input_width);

        Ok(Self { config, session, anchor_grid })
    }

    /// Run inference on a single video frame.
    ///
    /// Returns zero or more detections.  Returns an empty vec when no model
    /// is available.
    pub fn detect(
        &mut self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<Detection>, String> {
        let session = match &mut self.session {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };

        // ── 1. Pre-process ──────────────────────────────────────────
        let LetterBox { tensor, scale, pad_x, pad_y } = letterbox(
            frame,
            width,
            height,
            self.config.input_width,
            self.config.input_height,
        );

        // ── 2. Inference ────────────────────────────────────────────
        let array = ndarray::Array4::from_shape_vec(
            (
                1,
                3,
                self.config.input_height as usize,
                self.config.input_width as usize,
            ),
            tensor,
        )
        .map_err(|e| format!("tensor shape: {e}"))?;

        let input_tensor = ort::value::Tensor::from_array(array)
            .map_err(|e| format!("tensor from array: {e}"))?;

        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("inference failed: {e}"))?;

        // ── 3. Parse output ─────────────────────────────────────────
        let tensor_ref: ort::value::TensorRef<'_, f32> = outputs[0]
            .downcast_ref()
            .map_err(|e| format!("output downcast: {e}"))?;
        let (_shape, output_data) = tensor_ref
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("output data: {e}"))?;

        let num_classes = self.config.class_names.len();
        let num_predictions = self.anchor_grid.num_predictions;

        // Expected shape: [1, 4 + num_classes, num_predictions].
        let expected_channels = 4 + num_classes;
        if output_data.len() < expected_channels * num_predictions {
            return Err(format!(
                "Unexpected output size: got {} elements, expected at least {}",
                output_data.len(),
                expected_channels * num_predictions
            ));
        }

        let candidates = self.anchor_grid.decode(
            output_data,
            num_classes,
            self.config.conf_threshold,
            width,
            height,
            scale,
            pad_x,
            pad_y,
        );

        // ── 4. NMS ──────────────────────────────────────────────────
        let kept = non_max_suppression(candidates, self.config.iou_threshold);

        let detections: Vec<Detection> = kept
            .into_iter()
            .map(|(x1, y1, x2, y2, confidence, class_id)| Detection {
                x1,
                y1,
                x2,
                y2,
                confidence,
                class_id,
            })
            .collect();

        log::debug!("detect() returned {} detections", detections.len());
        Ok(detections)
    }

    pub fn config(&self) -> &YoloConfig {
        &self.config
    }

    pub fn is_model_available(&self) -> bool {
        self.session.is_some()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_constructs_without_model() {
        let cfg = YoloConfig {
            model_path: "nonexistent.onnx".into(),
            conf_threshold: 0.5,
            iou_threshold: 0.45,
            input_width: 640,
            input_height: 640,
            class_names: vec!["stop_sign".into()],
        };
        let detector = YoloDetector::new(cfg).expect("Constructor should not fail");
        assert!(!detector.is_model_available());
    }

    #[test]
    fn test_detect_returns_empty_when_no_model() {
        let cfg = YoloConfig {
            model_path: "nonexistent.onnx".into(),
            conf_threshold: 0.5,
            iou_threshold: 0.45,
            input_width: 640,
            input_height: 640,
            class_names: vec![],
        };
        let mut detector = YoloDetector::new(cfg).unwrap();
        let results = detector.detect(&[], 640, 480).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_anchor_grid_size() {
        let grid = AnchorGrid::new(640);
        assert_eq!(grid.num_predictions, 8400);
    }

    #[test]
    fn test_nms_keeps_best() {
        let boxes = vec![
            (10.0, 10.0, 100.0, 100.0, 0.9, 0u32),
            (15.0, 15.0, 95.0, 95.0, 0.8, 0u32),
            (200.0, 200.0, 300.0, 300.0, 0.7, 0u32),
        ];
        let kept = non_max_suppression(boxes, 0.5);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].4 - 0.9).abs() < 1e-6);
    }
}

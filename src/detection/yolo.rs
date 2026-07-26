//! YOLOv8 / YOLOv11 ONNX object detector.
//!
//! Loads an ONNX model via `ort`, pre-processes frames (letterbox, normalize,
//! HWC->CHW), runs inference, and post-processes (grid decode, sigmoid,
//! confidence filter, NMS, scale to original dimensions).
//!
//! ## Graceful degradation
//!
//! If no model file exists at the configured path, the detector returns empty
//! results so the pipeline can run for data collection or integration testing
//! before a custom model is trained.

use std::path::Path;

use crate::config::ModelConfig;

// ─────────────────────────────────────────────────────────────────────────────
//  Detection
// ─────────────────────────────────────────────────────────────────────────────

/// A single object detection produced by the YOLO model.
///
/// All coordinates are **absolute pixel values** in the original
/// (un-resized) image.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Left edge of the bounding box in pixels (inclusive).
    pub x1: f32,
    /// Top edge of the bounding box in pixels (inclusive).
    pub y1: f32,
    /// Right edge of the bounding box in pixels (inclusive).
    pub x2: f32,
    /// Bottom edge of the bounding box in pixels (inclusive).
    pub y2: f32,
    /// Detection confidence score in [0.0, 1.0].
    pub confidence: f32,
    /// Zero-based class index into the model's class list.
    pub class_id: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
//  YoloConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration parameters for the YOLO detector.
#[derive(Debug, Clone)]
pub struct YoloConfig {
    /// Path to the INT8-quantized ONNX model file.
    pub model_path: String,
    /// Minimum confidence in [0, 1]. Detections below this are discarded.
    pub conf_threshold: f32,
    /// NMS IoU threshold in [0, 1]. Overlapping boxes above this are suppressed.
    pub iou_threshold: f32,
    /// Width the model expects after letterbox resize (e.g. 640).
    pub input_width: u32,
    /// Height the model expects after letterbox resize (e.g. 640).
    pub input_height: u32,
    /// Ordered class names; index -> class_id.
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
    /// Scale factor applied to fit the original image into the model input.
    scale: f32,
    /// Horizontal padding added (in model-input pixel space).
    pad_x: f32,
    /// Vertical padding added (in model-input pixel space).
    pad_y: f32,
}

/// Resize `frame` to fit within `dst_w x dst_h` while preserving aspect ratio,
/// then pad with gray (114/255) to exactly `dst_w x dst_h`.
///
/// Returns a CHW float32 tensor (normalized to [0, 1]) plus the scale and
/// padding needed to map detections back to the original image.
///
/// * `frame` — Raw RGB8 pixel buffer, row-major, length `src_h * src_w * 3`.
/// * `src_w`, `src_h` — Dimensions of the source frame.
/// * `dst_w`, `dst_h` — Dimensions the model expects (e.g. 640 x 640).
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

    let src_img =
        image::RgbImage::from_raw(src_w, src_h, frame.to_vec()).expect("valid frame buffer");
    let resized = image::imageops::resize(
        &src_img,
        new_w,
        new_h,
        image::imageops::FilterType::CatmullRom,
    );

    // Initialise CHW float32 tensor with gray fill (114/255 ~= 0.447).
    let mut tensor = vec![114.0f32 / 255.0; (dst_w * dst_h * 3) as usize];
    let total = (dst_w * dst_h) as usize;

    // Copy resized pixels into the centre of the padded canvas.
    // CHW layout: channel 0 = R, 1 = G, 2 = B, each row-major.
    for y in 0..new_h {
        for x in 0..new_w {
            let pixel = resized.get_pixel(x, y);
            let canvas_x = (x as f32 + pad_x) as u32;
            let canvas_y = (y as f32 + pad_y) as u32;
            let idx = (canvas_y * dst_w + canvas_x) as usize;
            tensor[idx] = pixel[0] as f32 / 255.0;
            tensor[total + idx] = pixel[1] as f32 / 255.0;
            tensor[2 * total + idx] = pixel[2] as f32 / 255.0;
        }
    }

    LetterBox { tensor, scale, pad_x, pad_y }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bounding box helper
// ─────────────────────────────────────────────────────────────────────────────

/// An axis-aligned bounding box with associated confidence and class.
#[derive(Debug, Clone, Copy)]
struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
    class_id: u32,
}

/// Intersection-over-Union of two bounding boxes.
fn box_iou(a: &BBox, b: &BBox) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Non-maximum suppression
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy non-maximum suppression.
///
/// 1. Sorts candidates by descending confidence.
/// 2. Picks the highest-confidence box, suppresses all others with
///    IoU > `iou_threshold`.
/// 3. Repeats until no candidates remain.
///
/// * `candidates` — Unfiltered detections.
/// * `iou_threshold` — Box pairs with IoU above this value have the
///   lower-confidence member removed.
fn non_max_suppression(mut candidates: Vec<BBox>, iou_threshold: f32) -> Vec<BBox> {
    candidates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut active = vec![true; candidates.len()];

    for i in 0..candidates.len() {
        if !active[i] {
            continue;
        }
        keep.push(candidates[i]);

        for j in (i + 1)..candidates.len() {
            if active[j] && box_iou(&candidates[i], &candidates[j]) > iou_threshold {
                active[j] = false;
            }
        }
    }

    keep
}

// ─────────────────────────────────────────────────────────────────────────────
//  Grid / stride helpers for YOLOv8 decoding
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-computed anchor information for YOLOv8 output decoding.
///
/// YOLOv8 produces predictions at three strides (8, 16, 32). For a 640x640
/// input this yields 80x80 + 40x40 + 20x20 = 8400 anchors.
struct AnchorGrid {
    /// Per-anchor data: `(grid_x, grid_y, stride)`.
    anchors: Vec<(f32, f32, f32)>,
    /// Total number of anchors across all stride levels.
    num_predictions: usize,
}

impl AnchorGrid {
    /// Build the anchor grid for a square model input.
    ///
    /// * `input_size` — Width and height the model expects (e.g. 640).
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

    /// Decode raw YOLOv8 output tensor into candidate bounding boxes.
    ///
    /// The model's output is laid out as
    /// `[1, 4 + num_classes, num_predictions]` in CHW format (channel-major).
    /// For each anchor, channels 0–3 carry the bounding box (cx, cy, w, h),
    /// and channels 4..4+num_classes carry class logits.
    ///
    /// * `output` — Raw float32 data from the ONNX model output.
    /// * `num_classes` — Number of classes the model was trained on.
    /// * `conf_threshold` — Minimum confidence to keep a candidate.
    /// * `orig_w`, `orig_h` — Original frame dimensions (for coordinate
    ///   scaling).
    /// * `scale` — Scale factor from `letterbox()`.
    /// * `pad_x`, `pad_y` — Padding applied by `letterbox()`.
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
    ) -> Vec<BBox> {
        let stride_size = self.num_predictions;
        let mut candidates = Vec::new();

        for (i, &(grid_x, grid_y, stride)) in self.anchors.iter().enumerate() {
            // Raw bbox predictions at channel offsets 0..3.
            let cx_raw = output[i];
            let cy_raw = output[1 * stride_size + i];
            let w_raw = output[2 * stride_size + i];
            let h_raw = output[3 * stride_size + i];

            // Standard YOLOv8 grid-space decoding.
            let cx = (sigmoid(cx_raw) * 2.0 - 0.5 + grid_x) * stride;
            let cy = (sigmoid(cy_raw) * 2.0 - 0.5 + grid_y) * stride;
            let w = (sigmoid(w_raw) * 2.0).powi(2) * stride;
            let h = (sigmoid(h_raw) * 2.0).powi(2) * stride;

            // Find the class with the highest sigmoid score.
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

            // cxcywh -> xyxy in model-input coordinates.
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            // Remove padding and scale back to original image space.
            let x1_orig = ((x1 - pad_x) / scale).clamp(0.0, orig_w as f32);
            let y1_orig = ((y1 - pad_y) / scale).clamp(0.0, orig_h as f32);
            let x2_orig = ((x2 - pad_x) / scale).clamp(0.0, orig_w as f32);
            let y2_orig = ((y2 - pad_y) / scale).clamp(0.0, orig_h as f32);

            if (x2_orig - x1_orig) < 1.0 || (y2_orig - y1_orig) < 1.0 {
                continue;
            }

            candidates.push(BBox {
                x1: x1_orig,
                y1: y1_orig,
                x2: x2_orig,
                y2: y2_orig,
                confidence: best_conf,
                class_id: best_class,
            });
        }

        candidates
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Sigmoid
// ─────────────────────────────────────────────────────────────────────────────

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
    /// Loads the ONNX model if `config.model_path` exists on disk.  A missing
    /// model is **not** an error — the detector returns empty detections.
    ///
    /// * `config` — Detector settings (path, thresholds, input size, classes).
    pub fn new(config: YoloConfig) -> Result<Self, String> {
        let model_path = Path::new(&config.model_path);

        let session = if model_path.exists() {
            log::info!("Loading ONNX model from '{}'", config.model_path);
            let session = ort::session::Session::builder()
                .map_err(|e| format!("ort init: {e}"))?
                .commit_from_file(model_path)
                .map_err(|e| format!("Failed to load model '{}': {e}", config.model_path))?;
            log::info!(
                "Model loaded: {}x{} input",
                config.input_width,
                config.input_height
            );
            Some(session)
        } else {
            log::warn!(
                "ONNX model not found at '{}'. \
                 Detector returns empty results. \
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
    /// file is available.
    ///
    /// * `frame` — Flattened RGB8 pixel data, row-major, length `H * W * 3`.
    /// * `width` — Frame width in pixels.
    /// * `height` — Frame height in pixels.
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

        // ── 1. Pre-process: letterbox + normalise + HWC -> CHW ──────
        let LetterBox { tensor, scale, pad_x, pad_y } = letterbox(
            frame,
            width,
            height,
            self.config.input_width,
            self.config.input_height,
        );

        // ── 2. ONNX Runtime inference ───────────────────────────────
        let array = ndarray::Array4::from_shape_vec(
            (1, 3, self.config.input_height as usize, self.config.input_width as usize),
            tensor,
        )
        .map_err(|e| format!("tensor shape: {e}"))?;

        let input_tensor = ort::value::Tensor::from_array(array)
            .map_err(|e| format!("tensor from array: {e}"))?;

        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("inference failed: {e}"))?;

        // ── 3. Parse output tensor ──────────────────────────────────
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
                "Unexpected output: {} elements, expected at least {}",
                output_data.len(),
                expected_channels * num_predictions
            ));
        }

        // ── 4. Decode raw output -> candidates ──────────────────────
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

        // ── 5. Non-maximum suppression ───────────────────────────────
        let kept = non_max_suppression(candidates, self.config.iou_threshold);

        let detections: Vec<Detection> = kept
            .into_iter()
            .map(|b| Detection {
                x1: b.x1,
                y1: b.y1,
                x2: b.x2,
                y2: b.y2,
                confidence: b.confidence,
                class_id: b.class_id,
            })
            .collect();

        log::debug!("detect() returned {} detections", detections.len());
        Ok(detections)
    }

    /// Shared reference to the detector's configuration.
    pub fn config(&self) -> &YoloConfig {
        &self.config
    }

    /// Whether an ONNX model file was successfully loaded.
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

    /// Detector should construct without error when the model file is absent.
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

    /// Without a model, `detect()` should return an empty vec, not an error.
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

    /// For a 640x640 model input, YOLOv8 produces exactly 8400 anchors.
    #[test]
    fn test_anchor_grid_size() {
        let grid = AnchorGrid::new(640);
        assert_eq!(grid.num_predictions, 8400);
    }

    /// Two overlapping boxes should be reduced to one high-confidence box.
    #[test]
    fn test_nms_keeps_best() {
        let candidates = vec![
            BBox { x1: 10.0, y1: 10.0, x2: 100.0, y2: 100.0, confidence: 0.9, class_id: 0 },
            BBox { x1: 15.0, y1: 15.0, x2: 95.0, y2: 95.0, confidence: 0.8, class_id: 0 },
            BBox { x1: 200.0, y1: 200.0, x2: 300.0, y2: 300.0, confidence: 0.7, class_id: 0 },
        ];
        let kept = non_max_suppression(candidates, 0.5);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].confidence - 0.9).abs() < 1e-6);
    }
}

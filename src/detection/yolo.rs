//! YOLOv8 / YOLOv11 ONNX object detector.
//!
//! Loads an ONNX model via `ort`, pre-processes frames (letterbox, normalize,
//! HWC->CHW), runs inference, and post-processes (decode baked DFL boxes +
//! sigmoid class scores, confidence filter, NMS, scale to original dimensions).
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
fn letterbox(frame: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> LetterBox {
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

    let mut tensor = vec![114.0f32 / 255.0; (dst_w * dst_h * 3) as usize];
    let total = (dst_w * dst_h) as usize;

    for y in 0..new_h {
        for x in 0..new_w {
            let pixel = resized.get_pixel(x, y);
            let idx = ((y as f32 + pad_y) as u32 * dst_w + (x as f32 + pad_x) as u32) as usize;
            tensor[idx] = pixel[0] as f32 / 255.0;
            tensor[total + idx] = pixel[1] as f32 / 255.0;
            tensor[2 * total + idx] = pixel[2] as f32 / 255.0;
        }
    }

    LetterBox {
        tensor,
        scale,
        pad_x,
        pad_y,
    }
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
    let union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Non-maximum suppression
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy non-maximum suppression.
///
/// Sorts by descending confidence, picks the best, suppresses all others
/// with IoU > `iou_threshold`, repeats.
fn non_max_suppression(mut candidates: Vec<BBox>, iou_threshold: f32) -> Vec<BBox> {
    candidates.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; candidates.len()];
    let mut keep = Vec::new();

    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(candidates[i]);
        for j in (i + 1)..candidates.len() {
            if !suppressed[j] && box_iou(&candidates[i], &candidates[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

// ─────────────────────────────────────────────────────────────────────────────
//  Grid / stride helpers for YOLOv8 decoding
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-computed anchor-count metadata for YOLOv8 output decoding.
///
/// The Ultralytics ONNX export produces predictions at three strides
/// (8, 16, 32). For a 640x640 input this yields
/// 80x80 + 40x40 + 20x20 = 8400 predictions.
struct AnchorGrid {
    num_predictions: usize,
}

impl AnchorGrid {
    fn new(input_size: u32) -> Self {
        let num_predictions: usize = [8u32, 16, 32]
            .iter()
            .map(|&stride| {
                let grid = input_size / stride;
                (grid * grid) as usize
            })
            .sum();
        Self { num_predictions }
    }

    /// Decode raw YOLOv8 output tensor into candidate bounding boxes.
    ///
    /// The Ultralytics ONNX export (the format `scripts/download_test_model.sh`
    /// fetches) bakes the DFL box decode and class sigmoid into the graph.
    /// Each prediction is therefore already decoded: channels 0-3 hold a box
    /// `(cx, cy, w, h)` in model-input pixel space, and channels 4+ hold
    /// sigmoid-activated class probabilities. No grid offsets or sigmoid are
    /// applied here — values are read as-is and mapped from letterbox space
    /// back to the original frame.
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
        let stride = self.num_predictions;

        (0..self.num_predictions)
            .filter_map(|i| {
                let cx = output[i];
                let cy = output[stride + i];
                let w = output[2 * stride + i];
                let h = output[3 * stride + i];

                let (best_class, best_conf) = (0..num_classes)
                    .map(|c| (c as u32, output[(4 + c) * stride + i]))
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or((0, 0.0));

                (best_conf >= conf_threshold).then_some((
                    ((cx - w / 2.0 - pad_x) / scale).clamp(0.0, orig_w as f32),
                    ((cy - h / 2.0 - pad_y) / scale).clamp(0.0, orig_h as f32),
                    ((cx + w / 2.0 - pad_x) / scale).clamp(0.0, orig_w as f32),
                    ((cy + h / 2.0 - pad_y) / scale).clamp(0.0, orig_h as f32),
                    best_class,
                    best_conf,
                ))
            })
            .filter(|(x1, y1, x2, y2, _, _)| (x2 - x1) >= 1.0 && (y2 - y1) >= 1.0)
            .map(|(x1, y1, x2, y2, class_id, confidence)| BBox {
                x1,
                y1,
                x2,
                y2,
                confidence,
                class_id,
            })
            .collect()
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
    /// Loads the ONNX model if `config.model_path` exists on disk.  A missing
    /// model is **not** an error.
    pub fn new(config: YoloConfig) -> Result<Self, String> {
        let path = Path::new(&config.model_path);

        let session = match path.exists() {
            true => {
                log::info!("Loading ONNX model from '{}'", config.model_path);
                let s = ort::session::Session::builder()
                    .map_err(|e| format!("ort init: {e}"))?
                    .commit_from_file(path)
                    .map_err(|e| format!("Failed to load model '{}': {e}", config.model_path))?;
                log::info!(
                    "Model loaded: {}x{} input",
                    config.input_width,
                    config.input_height
                );
                Some(s)
            }
            false => {
                log::warn!(
                    "ONNX model not found at '{}'. Detector returns empty results. \
                     Train a model (see CLOUD_TRAINING.md) and place it at this path.",
                    config.model_path
                );
                None
            }
        };

        let anchor_grid = AnchorGrid::new(config.input_width);
        Ok(Self {
            config,
            session,
            anchor_grid,
        })
    }

    /// Run inference on a single video frame.
    ///
    /// Returns zero or more detections, or an empty vec when no model is
    /// available.
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

        let LetterBox {
            tensor,
            scale,
            pad_x,
            pad_y,
        } = letterbox(
            frame,
            width,
            height,
            self.config.input_width,
            self.config.input_height,
        );

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

        let input_tensor =
            ort::value::Tensor::from_array(array).map_err(|e| format!("tensor from array: {e}"))?;

        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("inference failed: {e}"))?;

        let tensor_ref: ort::value::TensorRef<'_, f32> = outputs[0]
            .downcast_ref()
            .map_err(|e| format!("output downcast: {e}"))?;

        let (_shape, output_data) = tensor_ref
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("output data: {e}"))?;

        let num_classes = self.config.class_names.len();
        let num_predictions = self.anchor_grid.num_predictions;

        let expected = 4 + num_classes;
        match output_data.len() >= expected * num_predictions {
            false => Err(format!(
                "Expected >= {} elements, got {}",
                expected * num_predictions,
                output_data.len()
            )),
            true => {
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
        }
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

    /// Regression test for the baked-decode ONNX export format.
    ///
    /// Channels 0-3 carry an already-decoded box `(cx, cy, w, h)` in
    /// model-input pixel space and channels 4+ carry sigmoid-activated class
    /// probabilities (no grid offsets, no extra sigmoid).
    #[test]
    fn test_decode_baked_export_format() {
        // 640 input -> 8400 predictions, 2 classes -> 6 channels per anchor.
        let grid = AnchorGrid::new(640);
        let n = grid.num_predictions;
        let num_classes = 2;
        let mut output = vec![0.0f32; (4 + num_classes) * n];

        // Anchor 100: box (cx=50, cy=60, w=20, h=10), class 1 @ 0.9.
        let i = 100;
        output[i] = 50.0; // cx
        output[n + i] = 60.0; // cy
        output[2 * n + i] = 20.0; // w
        output[3 * n + i] = 10.0; // h
        output[(4 + 1) * n + i] = 0.9; // class 1 probability

        let boxes = grid.decode(
            &output,
            num_classes,
            0.5, // conf threshold
            640,
            640,
            1.0, // scale
            0.0, // pad_x
            0.0, // pad_y
        );

        assert_eq!(boxes.len(), 1, "exactly one box should pass the threshold");
        let b = &boxes[0];
        assert_eq!(b.class_id, 1);
        assert!((b.confidence - 0.9).abs() < 1e-6);
        assert!((b.x1 - 40.0).abs() < 1e-4);
        assert!((b.y1 - 55.0).abs() < 1e-4);
        assert!((b.x2 - 60.0).abs() < 1e-4);
        assert!((b.y2 - 65.0).abs() < 1e-4);
    }

    #[test]
    fn test_nms_keeps_best() {
        let candidates = vec![
            BBox {
                x1: 10.0,
                y1: 10.0,
                x2: 100.0,
                y2: 100.0,
                confidence: 0.9,
                class_id: 0,
            },
            BBox {
                x1: 15.0,
                y1: 15.0,
                x2: 95.0,
                y2: 95.0,
                confidence: 0.8,
                class_id: 0,
            },
            BBox {
                x1: 200.0,
                y1: 200.0,
                x2: 300.0,
                y2: 300.0,
                confidence: 0.7,
                class_id: 0,
            },
        ];
        let kept = non_max_suppression(candidates, 0.5);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].confidence - 0.9).abs() < 1e-6);
    }
}

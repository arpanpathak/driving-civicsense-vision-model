//! # 🎯 YOLOv8 / YOLOv11 Object Detector
//!
//! Wraps an ONNX Runtime session for YOLO inference on edge hardware.
//! The model is quantized to INT8 for performance.
//!
//! ## Target Classes
//!
//! - `stop_sign`, `traffic_light`, `crosswalk`
//! - `vehicle`, `truck`, `bus`
//! - `intersection_zone`
//!
//! ## Graceful Degradation
//!
//! If no ONNX model file exists at the configured `model_path`, the detector
//! constructs successfully and returns **empty results**.  This enables the
//! data-collection and pipeline-testing workflow *before* a custom model is
//! trained — the pipeline runs, captures frames, and exercises every module
//! without a real model.
//!
//! ## Roadmap
//!
//! - [ ] Load ONNX model via `onnxruntime-rs`
//! - [ ] Letterbox resize to `(input_width × input_height)`
//! - [ ] Normalize to `[0, 1]`, transpose HWC → CHW
//! - [ ] Run `Session::run()`, parse output tensor
//! - [ ] Apply NMS with class-specific thresholds

use crate::config::ModelConfig;

// ─────────────────────────────────────────────────────────────────────────────
//  Detection
// ─────────────────────────────────────────────────────────────────────────────

/// A single object detection produced by the YOLO model.
///
/// Coordinates are **absolute pixel values** in the original (un-resized)
/// image.  The box convention is `(x1, y1)` = top-left corner,
/// `(x2, y2)` = bottom-right corner.
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

    /// Detection confidence score (0.0 – 1.0).
    ///
    /// Values closer to 1.0 indicate higher model certainty.
    pub confidence: f32,

    /// Zero-based class index into the model's class list.
    ///
    /// Mapping: 0 = stop_sign, 1 = traffic_light, 2 = crosswalk,
    /// 3 = vehicle, 4 = truck, 5 = bus, 6 = intersection_zone.
    pub class_id: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
//  YoloConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration parameters for the YOLO detector.
///
/// This struct is typically constructed from [`ModelConfig`] via the
/// [`From`] trait implementation below.
#[derive(Debug, Clone)]
pub struct YoloConfig {
    /// Filesystem path to the INT8-quantized ONNX model file.
    ///
    /// Example: `"weights/best-int8.onnx"`.  If the file does not exist
    /// the detector will return empty results (graceful degradation).
    pub model_path: String,

    /// Minimum confidence threshold (0.0 – 1.0).
    ///
    /// Detections with `confidence < conf_threshold` are discarded.
    pub conf_threshold: f32,

    /// NMS IoU threshold (0.0 – 1.0).
    ///
    /// Overlapping boxes with IoU > this value are suppressed.
    pub iou_threshold: f32,

    /// Width that the model expects after letterbox resize (e.g. 640).
    pub input_width: u32,

    /// Height that the model expects after letterbox resize (e.g. 640).
    pub input_height: u32,

    /// Ordered list of class names; index ↔ `class_id`.
    pub class_names: Vec<String>,
}

impl From<&ModelConfig> for YoloConfig {
    /// Converts the library-level [`ModelConfig`] into the detector-specific
    /// [`YoloConfig`].
    ///
    /// # Parameters
    /// - `cfg` — A reference to [`ModelConfig`] (from `config.rs`).
    ///
    /// # Returns
    /// A fully populated `YoloConfig`.
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
//  YoloDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Real-time YOLO object detector wrapping an (optional) ONNX Runtime session.
///
/// # Graceful degradation
///
/// If the model file specified in [`YoloConfig::model_path`] does not exist,
/// `detect()` returns `Ok(vec![])` — an empty vector.  This lets the rest
/// of the pipeline run for data-collection and integration testing.
pub struct YoloDetector {
    /// Detector configuration (path, thresholds, input size, class names).
    config: YoloConfig,

    /// Whether the ONNX model file exists on disk at construction time.
    model_available: bool,
}

impl YoloDetector {
    /// Constructs a new `YoloDetector`.
    ///
    /// Checks whether the ONNX file exists.  A missing model is **not** an
    /// error — the detector will simply return empty results.
    ///
    /// # Parameters
    /// - `config` — A [`YoloConfig`] specifying the model path, thresholds,
    ///   input dimensions, and class vocabulary.
    ///
    /// # Returns
    /// - `Ok(YoloDetector)` — always succeeds (missing model is allowed).
    /// - `Err(String)` — reserved for future validation errors.
    pub fn new(config: YoloConfig) -> Result<Self, String> {
        let model_available = std::path::Path::new(&config.model_path).exists();
        if !model_available {
            log::warn!(
                "ONNX model not found at '{}'. Detector will return empty results. \
                 Train a model (see CLOUD_TRAINING.md) and place it at this path.",
                config.model_path
            );
        } else {
            log::info!(
                "ONNX model found at '{}'. ONNX Runtime session loading not yet wired. \
                 TODO: integrate onnxruntime-rs crate.",
                config.model_path
            );
        }
        Ok(Self {
            config,
            model_available,
        })
    }

    /// Runs inference on a single video frame.
    ///
    /// # Parameters
    /// - `frame` — Flattened RGB8 pixel buffer, row-major, length `H × W × 3`.
    /// - `width` — Frame width in pixels.
    /// - `height` — Frame height in pixels.
    ///
    /// # Returns
    /// - `Ok(Vec<Detection>)` — zero or more detections.
    ///   Returns an empty vec when:
    ///   - No model file is available (graceful degradation).
    ///   - ONNX inference is not yet wired.
    /// - `Err(String)` — reserved for future runtime inference errors.
    ///
    /// # Panics
    /// Never panics.
    pub fn detect(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<Detection>, String> {
        if !self.model_available {
            return Ok(Vec::new());
        }

        // ── Pre-processing placeholder ────────────────────────────────
        // TODO: letterbox resize to (input_width × input_height)
        // TODO: normalize to [0, 1] range
        // TODO: transpose from HWC to CHW if required by ONNX model
        let _ = (frame, width, height);

        // ── Inference placeholder ─────────────────────────────────────
        // TODO: onnxruntime::Session::run()
        // TODO: Parse raw output tensor into Detection structs
        // TODO: Apply NMS (Non-Maximum Suppression) with iou_threshold

        log::debug!("detect() called — ONNX inference not yet wired. Returning empty result.");
        Ok(Vec::new())
    }

    /// Returns a shared reference to the detector's configuration.
    ///
    /// # Returns
    /// A `&YoloConfig` containing the current model path, thresholds, etc.
    pub fn config(&self) -> &YoloConfig {
        &self.config
    }

    /// Returns whether the ONNX model file was found on disk.
    ///
    /// # Returns
    /// `true` if the model file existed at construction time, `false` otherwise.
    pub fn is_model_available(&self) -> bool {
        self.model_available
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The detector should construct successfully even when the model
    /// file does not exist (graceful degradation).
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

    /// Without a model file, `detect()` must return an empty vec (not an error).
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
        let detector = YoloDetector::new(cfg).unwrap();
        let results = detector.detect(&[], 640, 480).unwrap();
        assert!(results.is_empty());
    }
}

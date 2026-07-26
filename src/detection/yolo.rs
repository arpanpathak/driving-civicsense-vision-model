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
//! ## On-Device Inference
//!
//! The actual ONNX session is loaded lazily. If no model file is found at the
//! configured path, the detector returns an empty result set (graceful
//! degradation). This lets you run the data-collection / pipeline-testing
//! workflow before training a custom model.

use crate::config::ModelConfig;

/// A single detection result.
#[derive(Debug, Clone)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u32,
}

/// Configuration for the YOLO detector.
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

/// Real-time YOLO object detector.
pub struct YoloDetector {
    config: YoloConfig,
    /// Whether the ONNX model file exists on disk.
    model_available: bool,
}

impl YoloDetector {
    /// Loads an ONNX model from the given path.
    ///
    /// If the model file does not exist, the detector still constructs
    /// successfully but will return empty detections. This allows the
    /// project to be used for data collection and pipeline testing before
    /// a custom model is trained.
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

    /// Runs inference on a single frame and returns detections.
    ///
    /// # Arguments
    /// * `frame` - Flattened RGB8 image, row-major, shape (H×W×3).
    /// * `width` - Image width in pixels.
    /// * `height` - Image height in pixels.
    ///
    /// Returns an empty vec if the model is not available (data-collection
    /// mode) or if inference is not yet wired to onnxruntime.
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

    /// Returns a reference to the detector config.
    pub fn config(&self) -> &YoloConfig {
        &self.config
    }

    /// Returns true if the model file was found at construction time.
    pub fn is_model_available(&self) -> bool {
        self.model_available
    }
}

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
        let detector = YoloDetector::new(cfg).unwrap();
        let results = detector.detect(&[], 640, 480).unwrap();
        assert!(results.is_empty());
    }
}

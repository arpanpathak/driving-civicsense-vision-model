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
//! ## TODO
//!
//! - [ ] Load ONNX model at construction
//! - [ ] Implement pre-processing (letterbox → 640×640, normalize)
//! - [ ] Implement post-processing (NMS, class filtering)
//! - [ ] Benchmark latency on Intel NUC / Jetson / RPi5

#![allow(unused_variables, dead_code)]

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

/// Real-time YOLO object detector.
pub struct YoloDetector {
    config: YoloConfig,
    // TODO: onnxruntime::Session
}

impl YoloDetector {
    /// Loads an ONNX model from the given path.
    pub fn new(config: YoloConfig) -> Result<Self, String> {
        todo!("Load ONNX model via onnxruntime-rs");
    }

    /// Runs inference on a single frame and returns detections.
    ///
    /// # Arguments
    /// * `frame` - Flattened RGB8 image, row-major, shape (H×W×3).
    /// * `width` - Image width in pixels.
    /// * `height` - Image height in pixels.
    pub fn detect(&self, frame: &[u8], width: u32, height: u32) -> Result<Vec<Detection>, String> {
        todo!("Preprocess → inference → NMS → return detections");
    }
}

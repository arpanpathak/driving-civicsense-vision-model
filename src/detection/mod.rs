//! 🔍 Object Detection Module
//!
//! Wraps YOLOv8 / YOLOv11 ONNX inference for real-time detection of
//! traffic signs, vehicles, and intersection zones on edge hardware.
//!
//! ## 📋 TODO
//!
//! - [ ] Load INT8 quantized ONNX model via onnxruntime-rs
//! - [ ] Implement letterbox resize to 640×640
//! - [ ] Add NMS with class-specific thresholds
//! - [ ] Profile FPS on target hardware (Raspberry Pi 5, Coral, Qualcomm AR1)

pub mod yolo;

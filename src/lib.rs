//! # 🚗 Driving-CivicSense-Vision-Model
//!
//! AI-driven auxiliary perception for intersection discipline and lane-awareness.
//! Processes forward-facing video to prevent intersection violations and
//! encourage proper lane etiquette.
//!
//! ## Architecture
//!
//! ```text
//! [Camera Frame] → [YOLO Detector] → [Deep SORT Tracker] → [Analysis Modules] → [Alerts]
//! ```
//!
//! ## Library Modules
//!
//! | Module | Path | Purpose |
//! |--------|------|---------|
//! | [`config`] | `src/config.rs` | YAML-based typed configuration |
//! | [`detection`] | `src/detection/` | YOLOv8 / YOLOv11 ONNX inference wrapper |
//! | [`tracking`] | `src/tracking/` | Deep SORT multi-object tracker |
//! | [`modules::intersection`] | `src/modules/intersection.rs` | Stop sign & occupancy logic |
//! | [`modules::lane_speed`] | `src/modules/lane_speed.rs` | Relative speed estimation |
//! | [`utils::geometry`] | `src/utils/geometry.rs` | Pinhole distance, IoU, filters |
//! | [`utils::visualization`] | `src/utils/visualization.rs` | Debug overlay rendering |
//!
//! ## Binary
//!
//! The [`civicsense` binary](../civicsense/index.html) (defined in `src/main.rs`)
//! provides two subcommands:
//!
//! - **`run`**, full detection → tracking → analysis → alert pipeline
//! - **`collect`**, frame capture for training-data collection
//!
//! ## Status
//!
//! **Pre-alpha.** All modules are functional stubs, the ONNX inference
//! backend is not yet wired (requires `onnxruntime-rs`), so the detector
//! returns empty results.  The data-collection pipeline works end-to-end
//! on Raspberry Pi with a camera module.

pub mod config;
pub mod detection;
pub mod modules;
pub mod tracking;
pub mod train;
pub mod utils;
pub mod video;

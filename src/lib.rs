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

/// Pure algebraic functions: stopping distance, clearance time,
/// intrusion time, Lipschitz bound, reaction-time distribution,
/// and class-aware monocular depth estimation.  All functions are
/// stateless, deterministic, and side-effect-free.
pub mod algebra;

/// Typed, YAML-based configuration for all pipeline parameters
/// (camera intrinsics, model paths, detection thresholds, etc.).
pub mod config;

/// Decision engine pipeline: severity-ordered composition of the
/// six kinematic rules in [`rules`].  See [`decision::evaluate_safety`].
pub mod decision;

/// YOLOv8 / YOLOv11 ONNX inference wrapper (pre-processing,
/// forward pass, NMS, post-processing).
pub mod detection;

/// Core data types: [`Detection`], [`EgoState`], [`LeadVehicle`],
/// [`LightState`], [`LanePosition`], [`WarningLevel`], and
/// [`ReactionProfile`].  All types are immutable.
pub mod models;

/// Analysis modules: intersection occupancy and lane-speed
/// estimation with hysteresis.
pub mod modules;

/// Decision rules: six single-responsibility functions
/// (`rule_red` through `rule_stale`) implementing the kinematic
/// criteria of the paper (Section IV).
pub mod rules;

/// Deep SORT multi-object tracker: Kalman filter prediction,
/// Hungarian (IoU) association, and track-lifecycle management.
pub mod tracking;

/// YOLO training orchestrator: dataset preparation, GPU training
/// launch, and ONNX model export/validation.
pub mod train;

/// Utility functions: pinhole-camera geometry (distance
/// estimation), IoU computation, low-pass filtering, and
/// visualisation helpers.
pub mod utils;

/// Video source classification and frame-capture helpers for
/// training-data collection.
pub mod video;

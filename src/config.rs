//! # ⚙️ Configuration
//!
//! Deserializes the YAML config file into a typed Rust struct.
//! Supports loading from a file path or using sensible defaults.
//!
//! The default [`Config::default()`] mirrors `configs/default.yaml` and
//! provides reasonable starting values for development on 1280×720 video
//! with a 640×640 YOLO model.

use serde::Deserialize;

// ─────────────────────────────────────────────────────────────────────────────
//  Structs
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level configuration for the entire perception pipeline.
///
/// Every sub-field has a `#[serde(default)]` so that a partial YAML file
/// only needs to specify values the user wants to override; the rest fall
/// back to the sensible defaults defined in the `Default` impls below.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// YOLO ONNX model configuration (path, thresholds, input size, classes).
    pub model: ModelConfig,

    /// Camera intrinsic parameters (focal length, resolution, framerate).
    pub camera: CameraConfig,

    /// Deep SORT multi-object tracker parameters (gating, lifespan, init).
    pub tracking: TrackingConfig,

    /// Intersection safety module configuration (stop signs, occupancy grid).
    pub intersection: IntersectionConfig,

    /// Lane-speed courtesy module configuration (differential threshold, hysteresis).
    pub lane_speed: LaneSpeedConfig,
}

/// YOLO ONNX model parameters.
///
/// Controls which ONNX file to load, at what confidence / IoU thresholds
/// to accept detections, the letterbox input resolution, and the class
/// label vocabulary the model was trained on.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Filesystem path to the INT8-quantized ONNX model file.
    ///
    /// Default: `"weights/best-int8.onnx"`.
    /// If the file does not exist at construction time the detector logs a
    /// warning and returns empty results, this allows data-collection /
    /// pipeline development before a custom model is trained.
    pub path: String,

    /// Minimum confidence score (0.0 – 1.0) for a detection to be kept.
    ///
    /// Detections with `confidence < conf_threshold` are discarded during
    /// post-processing.  Default: `0.5`.
    pub conf_threshold: f32,

    /// Non-Maximum Suppression IoU threshold (0.0 – 1.0).
    ///
    /// When two bounding boxes overlap with IoU > `iou_threshold` the one
    /// with the lower confidence is suppressed.  Default: `0.45`.
    pub iou_threshold: f32,

    /// Width in pixels that the model expects after letterbox resize.
    ///
    /// YOLOv8n / YOLOv11n typically expect 640. Default: `640`.
    pub input_width: u32,

    /// Height in pixels that the model expects after letterbox resize.
    ///
    /// Default: `640`.
    pub input_height: u32,

    /// Ordered list of class names the model was trained on.
    ///
    /// The index in this vector corresponds to `class_id` in
    /// [`Detection`](crate::detection::yolo::Detection).
    /// Default: `["stop_sign", "traffic_light", "crosswalk", "vehicle",
    ///           "truck", "bus", "intersection_zone"]`.
    pub classes: Vec<String>,
}

/// Camera intrinsic parameters.
///
/// These are used by the geometry utilities (pinhole distance estimation,
/// BEV projection) to convert pixel coordinates into real-world distances.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    /// Camera focal length in pixels.
    ///
    /// Typical values: 600 – 800 for a wide-angle dashcam.  Used in the
    /// pinhole distance formula `Z = (focal_length × real_width) / pixel_width`.
    /// Default: `650.0`.
    pub focal_length: f32,

    /// Width of the camera frame in pixels.
    ///
    /// Default: `1280` (1080p).
    pub frame_width: u32,

    /// Height of the camera frame in pixels.
    ///
    /// Default: `720` (1080p).
    pub frame_height: u32,

    /// Nominal framerate of the camera in frames-per-second.
    ///
    /// Used to compute `dt_secs` for velocity estimation and to pace the
    /// inference loop.  Default: `30`.
    pub fps: u32,
}

/// Deep SORT / BoT-SORT tracker parameters.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TrackingConfig {
    /// Maximum cosine-distance threshold for appearance-gated matching.
    ///
    /// Tracks and detections whose appearance feature vectors differ more
    /// than this distance are not allowed to match.  Currently reserved for
    /// future CNN-based Re-ID; IoU-only matching is used for now.
    /// Default: `0.2`.
    pub max_cosine_distance: f32,

    /// Maximum number of consecutive frames a track can go unmatched before
    /// it is considered dead and removed.
    ///
    /// Default: `30` (~1 second at 30 fps).
    pub max_age: u32,

    /// Minimum number of hits (matched frames) before a track is promoted
    /// from "tentative" to "confirmed".
    ///
    /// Confirmed tracks are returned in the output; tentative tracks can
    /// optionally be hidden.  Default: `3`.
    pub n_init: u32,
}

/// Intersection safety analysis parameters.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct IntersectionConfig {
    /// Distance in meters beyond which a stop sign is ignored.
    ///
    /// If a stop sign is detected further away than this value no alert is
    /// raised even if the ego speed is high.  Default: `50.0` m.
    pub stop_sign_warning_distance: f32,

    /// Ego speed in mph above which a stop sign warning is issued.
    ///
    /// If the ego vehicle is travelling slower than this, a detected stop
    /// sign is considered to be handled normally.  Default: `10.0` mph.
    pub stop_sign_warning_speed: f32,

    /// Ego speed in mph above which a blocked-intersection alert fires.
    ///
    /// Default: `15.0` mph.
    pub blocked_intersection_speed: f32,

    /// Vehicle occupancy (as a % of frame area) above which the forward
    /// view counts as a blocked intersection.
    ///
    /// Default: `30.0` %. Calibrated for a narrow-FoV dashcam; ultra-wide
    /// lenses keep occupancy far lower and need a smaller threshold.
    pub blocked_occupancy_threshold: f32,

    /// Distance in meters from the stop line at which a blocked-intersection
    /// alert becomes relevant.
    ///
    /// Default: `30.0` m.
    pub blocked_distance_to_stop: f32,

    /// Resolution of the Bird's Eye View occupancy grid in meters per cell.
    ///
    /// Default: `0.5` m/cell.
    pub grid_resolution: f32,

    /// How far ahead (in meters) the occupancy grid extends.
    ///
    /// Default: `20.0` m.
    pub grid_ahead_distance: f32,
}

/// Lane-speed courtesy analysis parameters.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LaneSpeedConfig {
    /// Minimum speed differential in mph between the right lane and the ego
    /// lane that triggers a "Merge Right" reminder.
    ///
    /// Default: `5.0` mph.
    pub speed_diff_threshold: f32,

    /// How many seconds the speed differential must persist (without
    /// dropping below the threshold) before an alert is emitted.
    ///
    /// This provides hysteresis and avoids brief, false alerts.
    /// Default: `3.0` s.
    pub hysteresis_seconds: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Default implementations
// ─────────────────────────────────────────────────────────────────────────────

impl Default for ModelConfig {
    /// Returns a sensible default set of model parameters targeting a
    /// YOLOv8n / YOLOv11n model trained on the 7 CivicSense classes.
    fn default() -> Self {
        Self {
            path: "weights/best-int8.onnx".into(),
            conf_threshold: 0.5,
            iou_threshold: 0.45,
            input_width: 640,
            input_height: 640,
            classes: vec![
                "stop_sign".into(),
                "traffic_light".into(),
                "crosswalk".into(),
                "vehicle".into(),
                "truck".into(),
                "bus".into(),
                "intersection_zone".into(),
            ],
        }
    }
}

impl Default for CameraConfig {
    /// Returns a default camera config for a typical 1080p dashcam
    /// (1280×720, 30 fps, 650 px focal length).
    fn default() -> Self {
        Self {
            focal_length: 650.0,
            frame_width: 1280,
            frame_height: 720,
            fps: 30,
        }
    }
}

impl Default for TrackingConfig {
    /// Returns tracking parameters tuned for highway driving:
    /// tracks survive up to ~1 s of occlusion and require 3 hits to confirm.
    fn default() -> Self {
        Self {
            max_cosine_distance: 0.2,
            max_age: 30,
            n_init: 3,
        }
    }
}

impl Default for IntersectionConfig {
    /// Returns intersection-safety parameters calibrated for US urban
    /// streets (stop signs visible from 50 m, blocked-box alert at 15 mph).
    fn default() -> Self {
        Self {
            stop_sign_warning_distance: 50.0,
            stop_sign_warning_speed: 10.0,
            blocked_intersection_speed: 15.0,
            blocked_occupancy_threshold: 30.0,
            blocked_distance_to_stop: 30.0,
            grid_resolution: 0.5,
            grid_ahead_distance: 20.0,
        }
    }
}

impl Default for LaneSpeedConfig {
    /// Returns lane-speed parameters: alert when the right lane is ≥5 mph
    /// faster for at least 3 seconds.
    fn default() -> Self {
        Self {
            speed_diff_threshold: 5.0,
            hysteresis_seconds: 3.0,
        }
    }
}

impl Default for Config {
    /// Returns a complete `Config` populated entirely from sub-config defaults.
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            camera: CameraConfig::default(),
            tracking: TrackingConfig::default(),
            intersection: IntersectionConfig::default(),
            lane_speed: LaneSpeedConfig::default(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Methods
// ─────────────────────────────────────────────────────────────────────────────

impl Config {
    /// Reads and parses a YAML configuration file from `path`.
    ///
    /// # Parameters
    /// - `path`, Filesystem path to a `.yaml` or `.yml` file whose
    ///   structure matches the [`Config`] struct.  Unknown keys are silently
    ///   ignored by serde; missing keys use their `Default` values.
    ///
    /// # Returns
    /// - `Ok(Config)` on successful read + parse.
    /// - `Err(String)` if the file cannot be read or the YAML is invalid.
    ///
    /// # Errors
    /// - IO errors (file not found, permissions) are surfaced as `Err`.
    /// - YAML parse errors (bad syntax, wrong types) are surfaced as `Err`.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let contents =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read config: {e}"))?;
        serde_yaml::from_str(&contents).map_err(|e| format!("Failed to parse config: {e}"))
    }

    /// Attempts to load a config from `path`; falls back to [`Config::default()`]
    /// on any error, logging a warning.
    ///
    /// This is the recommended entry point for production code because the
    /// pipeline can start even without a config file present.
    ///
    /// # Parameters
    /// - `path`, Filesystem path to the YAML config file.
    ///
    /// # Returns
    /// A fully populated `Config`, either from the file or from defaults.
    ///
    /// # Panics
    /// Never panics.
    pub fn load_or_default(path: &str) -> Self {
        Self::from_file(path).unwrap_or_else(|e| {
            log::warn!("Could not load config from '{path}': {e}. Using defaults.");
            Config::default()
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that every sub-config has a nonzero value in its default.
    #[test]
    fn test_config_defaults() {
        let cfg = Config::default();
        assert_eq!(cfg.model.input_width, 640);
        assert_eq!(cfg.camera.fps, 30);
        assert_eq!(cfg.tracking.max_age, 30);
        assert!(cfg.intersection.stop_sign_warning_distance > 0.0);
        assert!(cfg.lane_speed.hysteresis_seconds > 0.0);
    }

    /// Ensures a full YAML string round-trips correctly through serde.
    #[test]
    fn test_config_yaml_roundtrip() {
        let yaml = r#"
model:
  path: "test.onnx"
  conf_threshold: 0.7
  iou_threshold: 0.5
  input_width: 640
  input_height: 640
  classes:
    - stop_sign
camera:
  focal_length: 500
  frame_width: 1920
  frame_height: 1080
  fps: 60
tracking:
  max_cosine_distance: 0.3
  max_age: 50
  n_init: 5
intersection:
  stop_sign_warning_distance: 40.0
  stop_sign_warning_speed: 8.0
  blocked_intersection_speed: 12.0
  blocked_distance_to_stop: 25.0
  grid_resolution: 1.0
  grid_ahead_distance: 15.0
lane_speed:
  speed_diff_threshold: 3.0
  hysteresis_seconds: 2.0
"#;
        let cfg: Config = serde_yaml::from_str(yaml).expect("YAML should parse");
        assert_eq!(cfg.model.path, "test.onnx");
        assert_eq!(cfg.model.conf_threshold, 0.7);
        assert_eq!(cfg.camera.frame_width, 1920);
        assert_eq!(cfg.tracking.max_age, 50);
    }
}

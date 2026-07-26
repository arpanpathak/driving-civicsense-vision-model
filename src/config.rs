//! # ⚙️ Configuration
//!
//! Deserializes the YAML config file into a typed Rust struct.
//! Supports loading from a file path or using sensible defaults.

use serde::Deserialize;

/// Top-level configuration for the entire pipeline.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    pub camera: CameraConfig,
    pub tracking: TrackingConfig,
    pub intersection: IntersectionConfig,
    pub lane_speed: LaneSpeedConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub path: String,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    pub input_width: u32,
    pub input_height: u32,
    pub classes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    pub focal_length: f32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub fps: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TrackingConfig {
    pub max_cosine_distance: f32,
    pub max_age: u32,
    pub n_init: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct IntersectionConfig {
    pub stop_sign_warning_distance: f32,
    pub stop_sign_warning_speed: f32,
    pub blocked_intersection_speed: f32,
    pub blocked_distance_to_stop: f32,
    pub grid_resolution: f32,
    pub grid_ahead_distance: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LaneSpeedConfig {
    pub speed_diff_threshold: f32,
    pub hysteresis_seconds: f32,
}

// ── Default implementations ──────────────────────────────────────────────

impl Default for ModelConfig {
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
    fn default() -> Self {
        Self {
            max_cosine_distance: 0.2,
            max_age: 30,
            n_init: 3,
        }
    }
}

impl Default for IntersectionConfig {
    fn default() -> Self {
        Self {
            stop_sign_warning_distance: 50.0,
            stop_sign_warning_speed: 10.0,
            blocked_intersection_speed: 15.0,
            blocked_distance_to_stop: 30.0,
            grid_resolution: 0.5,
            grid_ahead_distance: 20.0,
        }
    }
}

impl Default for LaneSpeedConfig {
    fn default() -> Self {
        Self {
            speed_diff_threshold: 5.0,
            hysteresis_seconds: 3.0,
        }
    }
}

impl Default for Config {
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

impl Config {
    /// Loads config from a YAML file path. Falls back to defaults on error.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let contents =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read config: {e}"))?;
        serde_yaml::from_str(&contents).map_err(|e| format!("Failed to parse config: {e}"))
    }

    /// Loads from file if it exists, otherwise returns defaults.
    pub fn load_or_default(path: &str) -> Self {
        Self::from_file(path).unwrap_or_else(|e| {
            log::warn!("Could not load config from '{path}': {e}. Using defaults.");
            Config::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = Config::default();
        assert_eq!(cfg.model.input_width, 640);
        assert_eq!(cfg.camera.fps, 30);
        assert_eq!(cfg.tracking.max_age, 30);
        assert!(cfg.intersection.stop_sign_warning_distance > 0.0);
        assert!(cfg.lane_speed.hysteresis_seconds > 0.0);
    }

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

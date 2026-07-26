//! # ⚙️ Configuration
//!
//! Deserializes the YAML config file into a typed Rust struct.
//!
//! ## TODO
//!
//! - [ ] Add serde derives for YAML deserialization
//! - [ ] Implement config file loading with defaults
//! - [ ] Add CLI override support (clap)

#![allow(unused_variables, dead_code)]

/// Top-level configuration for the entire pipeline.
#[derive(Debug, Clone)]
pub struct Config {
    pub model: ModelConfig,
    pub camera: CameraConfig,
    pub tracking: TrackingConfig,
    pub intersection: IntersectionConfig,
    pub lane_speed: LaneSpeedConfig,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub path: String,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    pub input_width: u32,
    pub input_height: u32,
    pub classes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CameraConfig {
    pub focal_length: f32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub fps: u32,
}

#[derive(Debug, Clone)]
pub struct TrackingConfig {
    pub max_cosine_distance: f32,
    pub max_age: u32,
    pub n_init: u32,
}

#[derive(Debug, Clone)]
pub struct IntersectionConfig {
    pub stop_sign_warning_distance: f32,
    pub stop_sign_warning_speed: f32,
    pub blocked_intersection_speed: f32,
    pub blocked_distance_to_stop: f32,
    pub grid_resolution: f32,
    pub grid_ahead_distance: f32,
}

#[derive(Debug, Clone)]
pub struct LaneSpeedConfig {
    pub speed_diff_threshold: f32,
    pub hysteresis_seconds: f32,
}

impl Config {
    /// Loads config from a YAML file path.
    pub fn from_file(path: &str) -> Result<Self, String> {
        todo!("Parse YAML config file via serde_yaml");
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                path: "weights/best-int8.onnx".into(),
                conf_threshold: 0.5,
                iou_threshold: 0.45,
                input_width: 640,
                input_height: 640,
                classes: vec![
                    "stop_sign".into(), "traffic_light".into(), "crosswalk".into(),
                    "vehicle".into(), "truck".into(), "bus".into(), "intersection_zone".into(),
                ],
            },
            camera: CameraConfig {
                focal_length: 650.0,
                frame_width: 1280,
                frame_height: 720,
                fps: 30,
            },
            tracking: TrackingConfig {
                max_cosine_distance: 0.2,
                max_age: 30,
                n_init: 3,
            },
            intersection: IntersectionConfig {
                stop_sign_warning_distance: 50.0,
                stop_sign_warning_speed: 10.0,
                blocked_intersection_speed: 15.0,
                blocked_distance_to_stop: 30.0,
                grid_resolution: 0.5,
                grid_ahead_distance: 20.0,
            },
            lane_speed: LaneSpeedConfig {
                speed_diff_threshold: 5.0,
                hysteresis_seconds: 3.0,
            },
        }
    }
}

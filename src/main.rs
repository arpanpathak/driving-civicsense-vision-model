//! Driving-CivicSense binary entry point.
//!
//! Provides two subcommands:
//! - `run` — detection -> tracking -> analysis -> alert pipeline
//! - `collect` — frame capture for training-data collection

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use civicsense::config::Config;
use civicsense::detection::yolo::{YoloConfig, YoloDetector};
use civicsense::modules::intersection::IntersectionAnalyzer;
use civicsense::modules::lane_speed::LaneSpeedAnalyzer;
use civicsense::tracking::deep_sort::MultiObjectTracker;
use civicsense::utils::visualization;
use civicsense::video;

// ─────────────────────────────────────────────────────────────────────────────
//  CLI
// ─────────────────────────────────────────────────────────────────────────────

/// AI-driven auxiliary perception for intersection discipline and
/// lane-awareness — built in Rust.
#[derive(Parser)]
#[command(name = "civicsense", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Full detection -> tracking -> analysis -> alert pipeline on a single
    /// video source.
    Run {
        /// Input source: video file, image, directory, or "camera".
        #[arg(short, long, default_value = "0")]
        source: String,

        /// Path to YAML configuration file.
        #[arg(short, long, default_value = "configs/default.yaml")]
        config: String,

        /// If set, writes annotated frames to ./output/frame_*.jpg.
        #[arg(short, long)]
        visualize: bool,

        /// Ego-vehicle speed in mph (fallback when no GPS/OBD feed).
        #[arg(long, default_value = "0.0")]
        ego_speed: f32,
    },

    /// Captures frames from a source and saves them as JPEGs for YOLO
    /// training-data annotation.
    Collect {
        /// Input source: video file, image directory, or "camera".
        #[arg(short, long, default_value = "0")]
        source: String,

        /// Directory where captured JPEG frames will be saved.
        #[arg(short, long, default_value = "data/raw")]
        output: String,

        /// Target frame-capture rate in fps (time-throttled).
        #[arg(short, long, default_value_t = 2.0)]
        rate: f32,

        /// Maximum frames to save (0 = unlimited).
        #[arg(short = 'n', long, default_value_t = 0)]
        max_frames: u64,

        /// Path to YAML config (used for camera intrinsics).
        #[arg(short, long, default_value = "configs/default.yaml")]
        config: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
//  Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .format_timestamp_millis()
    .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run { source, config, visualize, ego_speed } => {
            if let Err(e) = run_pipeline(&source, &config, visualize, ego_speed) {
                log::error!("Pipeline failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Collect { source, output, rate, max_frames, config: _config } => {
            if let Err(e) = collect_data(&source, &output, rate, max_frames) {
                log::error!("Data collection failed: {e}");
                std::process::exit(1);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Runs the full perception pipeline on a single video source.
fn run_pipeline(
    source: &str,
    config_path: &str,
    visualize: bool,
    ego_speed: f32,
) -> Result<(), String> {
    let config = Config::load_or_default(config_path);
    log::info!("Config loaded. Model: {}", config.model.path);

    let detector = YoloDetector::new(YoloConfig::from(&config.model))?;
    let mut tracker = MultiObjectTracker::new(
        config.tracking.max_age,
        config.tracking.n_init,
        config.tracking.max_cosine_distance,
    );

    let (mut frame_iter, frame_width, frame_height) =
        video::open_source(source, config.camera.frame_width, config.camera.frame_height)?;

    let mut intersection_analyzer = IntersectionAnalyzer::new(&config, frame_width, frame_height);
    let mut lane_speed_analyzer = LaneSpeedAnalyzer::new(&config);

    log::info!(
        "Pipeline started. Source: {source}, Resolution: {frame_width}x{frame_height}, Visualize: {visualize}"
    );
    log::info!(
        "Model available: {} — detections will be empty until an ONNX model is placed at '{}'",
        detector.is_model_available(),
        config.model.path
    );

    let mut frame_count: u64 = 0;
    let viz_output_dir = PathBuf::from("output");
    if visualize {
        std::fs::create_dir_all(&viz_output_dir)
            .map_err(|e| format!("Cannot create output dir: {e}"))?;
    }

    loop {
        let frame_data = match frame_iter() {
            Some(data) => data,
            None => {
                log::info!("End of video source. Processed {frame_count} frames.");
                break;
            }
        };

        let (frame_buffer, _frame_index) = frame_data;
        let dt_secs = 1.0 / config.camera.fps as f32;

        let detections = detector.detect(&frame_buffer, frame_width, frame_height)?;
        if !detections.is_empty() {
            log::debug!("Frame {frame_count}: {} detections", detections.len());
        }

        let tracks = tracker.update(&detections);
        let intersection_alerts =
            intersection_analyzer.analyze(&detections, ego_speed, dt_secs);
        let lane_alerts = lane_speed_analyzer.analyze(&tracks, ego_speed, dt_secs);

        log_intersection_alerts(&intersection_alerts);
        log_lane_alerts(&lane_alerts);

        if visualize && !detections.is_empty() {
            let mut viz_frame = frame_buffer.to_vec();
            let class_names = config.model.classes.clone();

            visualization::draw_detections(
                &mut viz_frame,
                frame_width,
                frame_height,
                &detections,
                &class_names,
            );

            if !intersection_alerts.is_empty() {
                visualization::draw_alert_text(
                    &mut viz_frame,
                    frame_width,
                    frame_height,
                    "STOP SIGN VIOLATION",
                );
            }
            if !lane_alerts.is_empty() {
                visualization::draw_alert_text(
                    &mut viz_frame,
                    frame_width,
                    frame_height,
                    "MERGE RIGHT REMINDER",
                );
            }

            let out_path = viz_output_dir.join(format!("frame_{:06}.jpg", frame_count));
            if let Err(e) = video::save_frame(&viz_frame, frame_width, frame_height, &out_path) {
                log::warn!("Failed to save visualization frame: {e}");
            }
        }

        frame_count += 1;

        if frame_count >= 300 && !visualize {
            log::info!("Processed {frame_count} frames (dev limit). Pass --visualize for output.");
            break;
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Alert logging
// ─────────────────────────────────────────────────────────────────────────────

fn log_intersection_alerts(alerts: &[civicsense::modules::intersection::IntersectionAlert]) {
    use civicsense::modules::intersection::IntersectionAlert;

    for alert in alerts {
        match alert {
            IntersectionAlert::StopSignViolation {
                confidence,
                distance_to_stop_line,
                ego_speed,
            } => {
                log::warn!(
                    "STOP SIGN VIOLATION! conf={:.2}, dist={:.1}ft, speed={:.1}mph",
                    confidence,
                    distance_to_stop_line,
                    ego_speed
                );
            }
            IntersectionAlert::BlockedIntersection {
                confidence,
                occupancy_pct,
                distance_to_stop_line,
                ego_speed,
            } => {
                log::warn!(
                    "BLOCKED INTERSECTION! conf={:.2}, occupancy={:.1}%, dist={:.1}ft, speed={:.1}mph",
                    confidence,
                    occupancy_pct,
                    distance_to_stop_line,
                    ego_speed
                );
            }
        }
    }
}

fn log_lane_alerts(alerts: &[civicsense::modules::lane_speed::LaneSpeedAlert]) {
    for alert in alerts {
        log::warn!(
            "MERGE RIGHT REMINDER! Right lane is {:.1} mph faster (for {:.1}s)",
            alert.speed_diff_mph,
            alert.duration_secs
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Data collection
// ─────────────────────────────────────────────────────────────────────────────

/// Captures frames from a source and saves timestamped JPEGs.
fn collect_data(
    source: &str,
    output_dir: &str,
    rate: f32,
    max_frames: u64,
) -> Result<(), String> {
    let output_path = PathBuf::from(output_dir);
    std::fs::create_dir_all(&output_path)
        .map_err(|e| format!("Cannot create output dir '{output_dir}': {e}"))?;

    let (mut frame_iter, frame_width, frame_height) =
        video::open_source(source, 1280, 720)?;

    log::info!(
        "Data collection started. Source: {source} -> Output: {output_dir}/"
    );
    log::info!(
        "Resolution: {frame_width}x{frame_height}, Rate: {rate} fps, Max frames: {}",
        if max_frames == 0 { "unlimited".into() } else { max_frames.to_string() }
    );

    let min_interval_ms = if rate > 0.0 {
        (1000.0 / rate) as u64
    } else {
        0
    };

    let start = Instant::now();
    let mut saved_count: u64 = 0;
    let mut last_save = Instant::now()
        .checked_sub(std::time::Duration::from_secs(3600))
        .unwrap_or(Instant::now());

    loop {
        let frame_data = match frame_iter() {
            Some(data) => data,
            None => {
                log::info!("End of source. Frames captured: {saved_count}");
                break;
            }
        };

        let (frame_buffer, _frame_index) = frame_data;

        let elapsed_since_last = last_save.elapsed().as_millis() as u64;
        if elapsed_since_last >= min_interval_ms {
            let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f");
            let filename = format!("capture_{}_{:06}.jpg", timestamp, saved_count);
            let out_path = output_path.join(&filename);

            if let Err(e) = video::save_frame(&frame_buffer, frame_width, frame_height, &out_path)
            {
                log::warn!("Failed to save frame: {e}");
            } else {
                log::info!("Saved: {}", out_path.display());
                saved_count += 1;
                last_save = Instant::now();

                if max_frames > 0 && saved_count >= max_frames {
                    log::info!("Reached max frames ({max_frames}). Stopping.");
                    break;
                }
            }
        }
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let effective_fps = if elapsed_secs > 0.0 {
        saved_count as f64 / elapsed_secs
    } else {
        0.0
    };

    log::info!(
        "Data collection complete. {saved_count} frames saved in {elapsed:.1?} ({effective_fps:.1} fps avg)"
    );

    Ok(())
}

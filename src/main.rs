//! Driving-CivicSense binary entry point.
//!
//! Subcommands:
//! - `run`, detection -> tracking -> analysis -> alert pipeline
//! - `collect`, frame capture for training-data collection
//! - `train`, YOLO training orchestrator (dataset prep, training, ONNX validation)

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use civicsense::config::Config;
use civicsense::detection::yolo::{YoloConfig, YoloDetector};
use civicsense::modules::intersection::{IntersectionAlert, IntersectionAnalyzer};
use civicsense::modules::lane_speed::{LaneSpeedAlert, LaneSpeedAnalyzer};
use civicsense::tracking::deep_sort::MultiObjectTracker;
use civicsense::train::{Dataset, TrainingRun};
use civicsense::utils::visualization;
use civicsense::video;

// ─────────────────────────────────────────────────────────────────────────────
//  CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "civicsense", version, about = "AI-driven auxiliary perception")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detection -> tracking -> analysis -> alert pipeline on a single source.
    Run {
        #[arg(short, long, default_value = "0")]
        source: String,
        #[arg(short, long, default_value = "configs/default.yaml")]
        config: String,
        #[arg(short, long)]
        visualize: bool,
        #[arg(long, default_value = "0.0")]
        ego_speed: f32,
    },
    /// Capture frames from a source and save as JPEGs for YOLO training data.
    Collect {
        #[arg(short, long, default_value = "0")]
        source: String,
        #[arg(short, long, default_value = "data/raw")]
        output: String,
        #[arg(short, long, default_value_t = 2.0)]
        rate: f32,
        #[arg(short = 'n', long, default_value_t = 0)]
        max_frames: u64,
        #[arg(short, long, default_value = "configs/default.yaml")]
        config: String,
    },
    /// YOLO model training: prepare dataset, train on GPU, validate ONNX.
    #[command(subcommand)]
    Train(TrainCommand),
}

#[derive(Subcommand)]
enum TrainCommand {
    /// Validate dataset structure, count images, check label formats.
    Prepare {
        /// Root directory of the labelled dataset.
        #[arg(short, long, default_value = "data/civicsense")]
        dataset: String,
        /// Split a flat directory of labelled images into train/val.
        #[arg(long)]
        split: Option<String>,
        /// Fraction of data to use for validation (0.0 – 0.5).
        #[arg(long, default_value_t = 0.2)]
        val_fraction: f64,
        /// Output YAML config path.
        #[arg(short, long, default_value = "configs/dataset.yaml")]
        output: String,
    },
    /// Run YOLO training + ONNX export on a GPU cloud instance.
    Run {
        /// Dataset YAML config path.
        #[arg(short, long, default_value = "configs/dataset.yaml")]
        data: String,
        /// Pretrained model (e.g. "yolov8n.pt" or a local path).
        #[arg(short, long, default_value = "yolov8n.pt")]
        model: String,
        /// Number of training epochs.
        #[arg(short, long, default_value_t = 100)]
        epochs: u32,
        /// Batch size.
        #[arg(short, long, default_value_t = 32)]
        batch: u32,
        /// Input image size.
        #[arg(long, default_value_t = 640)]
        imgsz: u32,
        /// GPU device(s).
        #[arg(long, default_value = "0")]
        device: String,
        /// Output project directory.
        #[arg(short, long, default_value = "runs/train")]
        project: String,
        /// Experiment name.
        #[arg(short, long, default_value = "civicsense")]
        name: String,
    },
    /// Load and validate an exported ONNX model with ort.
    Validate {
        /// Path to the ONNX model file.
        #[arg(short, long, default_value = "runs/train/civicsense/weights/best.onnx")]
        model: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
//  Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    match Cli::parse().command {
        Commands::Run {
            source,
            config,
            visualize,
            ego_speed,
        } => {
            let result =
                Pipeline::new(&source, &config, visualize, ego_speed).and_then(|mut p| p.run());
            if let Err(e) = result {
                log::error!("Pipeline failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Collect {
            source,
            output,
            rate,
            max_frames,
            ..
        } => {
            let result =
                Collector::new(&source, &output, rate, max_frames).and_then(|mut c| c.run());
            if let Err(e) = result {
                log::error!("Data collection failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Train(cmd) => match cmd {
            TrainCommand::Prepare {
                dataset,
                split,
                val_fraction,
                output,
            } => {
                let result = run_train_prepare(&dataset, split.as_deref(), val_fraction, &output);
                if let Err(e) = result {
                    log::error!("Dataset preparation failed: {e}");
                    std::process::exit(1);
                }
            }
            TrainCommand::Run {
                data,
                model,
                epochs,
                batch,
                imgsz,
                device,
                project,
                name,
            } => {
                let result = run_train_run(
                    &data, &model, epochs, batch, imgsz, &device, &project, &name,
                );
                if let Err(e) = result {
                    log::error!("Training failed: {e}");
                    std::process::exit(1);
                }
            }
            TrainCommand::Validate { model } => {
                let result = run_train_validate(&model);
                if let Err(e) = result {
                    log::error!("Model validation failed: {e}");
                    std::process::exit(1);
                }
            }
        },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Train subcommand implementations
// ─────────────────────────────────────────────────────────────────────────────

/// `civicsense train prepare`: validate/split dataset, write YAML config.
fn run_train_prepare(
    dataset: &str,
    split: Option<&str>,
    val_fraction: f64,
    output: &str,
) -> Result<(), String> {
    let dataset_path = std::path::Path::new(dataset);

    if let Some(source) = split {
        log::info!("Splitting flat directory '{}' -> '{}'", source, dataset);
        let ds = Dataset::split(std::path::Path::new(source), dataset_path, val_fraction)?;
        ds.write_yaml(std::path::Path::new(output))?;
        log::info!("Dataset config written to '{}'", output);
    } else {
        log::info!("Validating dataset at '{}'", dataset);
        let ds = Dataset::open(dataset_path)?;
        ds.write_yaml(std::path::Path::new(output))?;
        log::info!(
            "Dataset validated. {} train + {} val images. Config written to '{}'",
            ds.train_count,
            ds.val_count,
            output
        );
    }

    Ok(())
}

/// `civicsense train run`: train YOLO on GPU, export ONNX, validate.
#[allow(clippy::too_many_arguments)] // CLI plumbing; grouped config would obscure the subcommand
fn run_train_run(
    data: &str,
    model: &str,
    epochs: u32,
    batch: u32,
    imgsz: u32,
    device: &str,
    project: &str,
    name: &str,
) -> Result<(), String> {
    let run = TrainingRun {
        data_yaml: std::path::PathBuf::from(data),
        model: model.to_string(),
        epochs,
        batch,
        imgsz,
        device: device.to_string(),
        project: std::path::PathBuf::from(project),
        name: name.to_string(),
    };

    let onnx_path = run.run()?;
    log::info!("Trained model: {:?}", onnx_path);
    log::info!("Copy to weights/ and run: civicsense run --source test_video.mp4 --visualize");
    Ok(())
}

/// `civicsense train validate`: quick sanity check on an ONNX model.
fn run_train_validate(model: &str) -> Result<(), String> {
    civicsense::train::validate_onnx(std::path::Path::new(model))
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Orchestrates the perception pipeline for one video source.
struct Pipeline {
    detector: YoloDetector,
    tracker: MultiObjectTracker,
    intersection_analyzer: IntersectionAnalyzer,
    lane_speed_analyzer: LaneSpeedAnalyzer,
    frame_iter: video::FrameIter,
    config: Config,
    frame_width: u32,
    frame_height: u32,
    frame_count: u64,
    visualize: bool,
    ego_speed: f32,
    viz_output_dir: PathBuf,
}

impl Pipeline {
    fn new(
        source: &str,
        config_path: &str,
        visualize: bool,
        ego_speed: f32,
    ) -> Result<Self, String> {
        let config = Config::load_or_default(config_path);
        log::info!("Config loaded. Model: {}", config.model.path);

        let detector = YoloDetector::new(YoloConfig::from(&config.model))?;
        let tracker = MultiObjectTracker::new(
            config.tracking.max_age,
            config.tracking.n_init,
            config.tracking.max_cosine_distance,
        );

        let (frame_iter, frame_width, frame_height) = video::open_source(
            source,
            config.camera.frame_width,
            config.camera.frame_height,
        )?;

        let intersection_analyzer = IntersectionAnalyzer::new(&config, frame_width, frame_height);
        let lane_speed_analyzer = LaneSpeedAnalyzer::new(&config);

        let viz_output_dir = PathBuf::from("output");
        if visualize {
            std::fs::create_dir_all(&viz_output_dir)
                .map_err(|e| format!("Cannot create output dir: {e}"))?;
        }

        log::info!(
            "Pipeline started. Source: {source}, Resolution: {frame_width}x{frame_height}, Visualize: {visualize}"
        );
        log::info!("Model available: {}", detector.is_model_available());

        Ok(Self {
            detector,
            tracker,
            intersection_analyzer,
            lane_speed_analyzer,
            frame_iter,
            config,
            frame_width,
            frame_height,
            frame_count: 0,
            visualize,
            ego_speed,
            viz_output_dir,
        })
    }

    fn run(&mut self) -> Result<(), String> {
        loop {
            match (self.frame_iter)() {
                None => {
                    log::info!(
                        "End of video source. Processed {} frames.",
                        self.frame_count
                    );
                    return Ok(());
                }
                Some((buffer, _)) => {
                    if !self.process_frame(&buffer)? {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// Process a single frame. Returns `Ok(true)` to continue, `Ok(false)` to
    /// stop gracefully, or `Err` on failure.
    fn process_frame(&mut self, frame_buffer: &[u8]) -> Result<bool, String> {
        let dt_secs = 1.0 / self.config.camera.fps as f32;
        let detections = self
            .detector
            .detect(frame_buffer, self.frame_width, self.frame_height)?;
        let tracks = self.tracker.update(&detections);

        let intersection_alerts =
            self.intersection_analyzer
                .analyze(&detections, self.ego_speed, dt_secs);
        let lane_alerts = self
            .lane_speed_analyzer
            .analyze(&tracks, self.ego_speed, dt_secs);

        log_intersection_alerts(&intersection_alerts);
        log_lane_alerts(&lane_alerts);

        if self.visualize && !detections.is_empty() {
            self.render_frame(
                &detections,
                &intersection_alerts,
                &lane_alerts,
                frame_buffer,
            );
        }

        self.frame_count += 1;

        log::info!(
            "Frame {}: {} detections, {} tracks",
            self.frame_count,
            detections.len(),
            tracks.len(),
        );

        Ok(true)
    }

    /// Draw detections and alerts onto a frame buffer and save to disk.
    fn render_frame(
        &self,
        detections: &[civicsense::detection::yolo::Detection],
        intersection_alerts: &[IntersectionAlert],
        lane_alerts: &[LaneSpeedAlert],
        frame_buffer: &[u8],
    ) {
        let mut viz = frame_buffer.to_vec();
        let class_names = self.config.model.classes.clone();

        visualization::draw_detections(
            &mut viz,
            self.frame_width,
            self.frame_height,
            detections,
            &class_names,
        );

        if !intersection_alerts.is_empty() {
            visualization::draw_alert_text(
                &mut viz,
                self.frame_width,
                self.frame_height,
                "STOP SIGN VIOLATION",
            );
        }
        if !lane_alerts.is_empty() {
            visualization::draw_alert_text(
                &mut viz,
                self.frame_width,
                self.frame_height,
                "MERGE RIGHT REMINDER",
            );
        }

        let out_path = self
            .viz_output_dir
            .join(format!("frame_{:06}.jpg", self.frame_count));
        if let Err(e) = video::save_frame(&viz, self.frame_width, self.frame_height, &out_path) {
            log::warn!("Failed to save visualization frame: {e}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Collector
// ─────────────────────────────────────────────────────────────────────────────

/// Captures frames from a source and saves timestamped JPEGs for training data.
struct Collector {
    frame_iter: video::FrameIter,
    frame_width: u32,
    frame_height: u32,
    output_path: PathBuf,
    min_interval_ms: u64,
    max_frames: u64,
}

impl Collector {
    fn new(source: &str, output_dir: &str, rate: f32, max_frames: u64) -> Result<Self, String> {
        let output_path = PathBuf::from(output_dir);
        std::fs::create_dir_all(&output_path)
            .map_err(|e| format!("Cannot create output dir '{output_dir}': {e}"))?;

        let (frame_iter, frame_width, frame_height) = video::open_source(source, 1280, 720)?;

        let min_interval_ms = if rate > 0.0 {
            (1000.0 / rate) as u64
        } else {
            0
        };

        log::info!("Data collection started. Source: {source} -> {output_dir}/");
        log::info!(
            "Resolution: {frame_width}x{frame_height}, Rate: {rate} fps, Max frames: {}",
            if max_frames == 0 {
                "unlimited".into()
            } else {
                max_frames.to_string()
            }
        );

        Ok(Self {
            frame_iter,
            frame_width,
            frame_height,
            output_path,
            min_interval_ms,
            max_frames,
        })
    }

    fn run(&mut self) -> Result<(), String> {
        let start = Instant::now();
        let mut saved_count: u64 = 0;
        let mut last_save = Instant::now()
            .checked_sub(std::time::Duration::from_secs(3600))
            .unwrap_or(Instant::now());

        loop {
            let frame_buffer = match (self.frame_iter)() {
                Some((buf, _)) => buf,
                None => {
                    log::info!("End of source. Frames captured: {saved_count}");
                    break;
                }
            };

            if last_save.elapsed().as_millis() as u64 >= self.min_interval_ms
                && self.save_one_frame(&frame_buffer, saved_count).is_ok()
            {
                saved_count += 1;
                last_save = Instant::now();

                if self.max_frames > 0 && saved_count >= self.max_frames {
                    log::info!("Reached max frames ({}). Stopping.", self.max_frames);
                    break;
                }
            }
        }

        let elapsed = start.elapsed();
        let effective_fps = if elapsed.as_secs_f64() > 0.0 {
            saved_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        log::info!(
            "Data collection complete. {saved_count} frames saved in {elapsed:.1?} ({effective_fps:.1} fps avg)"
        );
        Ok(())
    }

    fn save_one_frame(&self, buffer: &[u8], index: u64) -> Result<(), String> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f");
        let filename = format!("capture_{}_{:06}.jpg", timestamp, index);
        let path = self.output_path.join(&filename);

        video::save_frame(buffer, self.frame_width, self.frame_height, &path).map_err(|e| {
            log::warn!("Failed to save frame: {e}");
            e
        })?;

        log::info!("Saved: {}", path.display());
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Alert logging
// ─────────────────────────────────────────────────────────────────────────────

fn log_intersection_alerts(alerts: &[IntersectionAlert]) {
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

fn log_lane_alerts(alerts: &[LaneSpeedAlert]) {
    for alert in alerts {
        log::warn!(
            "MERGE RIGHT REMINDER! Right lane is {:.1} mph faster (for {:.1}s)",
            alert.speed_diff_mph,
            alert.duration_secs
        );
    }
}

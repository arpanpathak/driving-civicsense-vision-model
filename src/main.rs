//! # 🚀 Driving-CivicSense — Entry Point
//!
//! CLI binary that orchestrates the perception pipeline or captures training
//! data from a camera or video source.
//!
//! ## Subcommands
//!
//! | Command | Description |
//! |---------|-------------|
//! | `run` | Detection → tracking → analysis → alert pipeline |
//! | `collect` | Frame capture for training-data collection |
//!
//! ## Usage
//!
//! ```bash
//! # Run the pipeline on a video file (macOS / development)
//! cargo run --bin civicsense -- run --source video.mp4 --visualize
//!
//! # Collect training data from head-mounted Pi camera
//! cargo run --bin civicsense -- collect --source camera --output ./data/raw/ --rate 2
//! ```

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use civicsense::config::Config;
use civicsense::detection::yolo::{YoloConfig, YoloDetector};
use civicsense::modules::intersection::IntersectionAnalyzer;
use civicsense::modules::lane_speed::LaneSpeedAnalyzer;
use civicsense::tracking::deep_sort::MultiObjectTracker;
use civicsense::utils::visualization;

// ─────────────────────────────────────────────────────────────────────────────
//  CLI
// ─────────────────────────────────────────────────────────────────────────────

/// Driving-CivicSense: AI-driven auxiliary perception for intersection
/// discipline and lane-awareness — built in Rust.
#[derive(Parser)]
#[command(name = "civicsense", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Runs the full detection → tracking → analysis → alert pipeline on a
    /// single video source.
    Run {
        /// Input source: path to a video/image/directory file, or `"camera"`
        /// for live capture from a Raspberry Pi camera module.
        ///
        /// Defaults to `"0"`, which tries camera mode.
        #[arg(short, long, default_value = "0")]
        source: String,

        /// Path to the YAML configuration file.
        ///
        /// Falls back to built-in defaults if the file does not exist.
        #[arg(short, long, default_value = "configs/default.yaml")]
        config: String,

        /// If set, writes annotated frames to `./output/frame_*.jpg` for
        /// visual debugging.  Also enables longer processing (no dev limit).
        #[arg(short, long)]
        visualize: bool,

        /// Ego-vehicle speed in mph (used when no real GPS/OBD feed is
        /// available, e.g. when processing pre-recorded video).
        #[arg(long, default_value = "0.0")]
        ego_speed: f32,
    },

    /// Captures frames from a video/camera source and saves them as JPEG
    /// images for YOLO training-data annotation.
    Collect {
        /// Input source: video file, image directory, or `"camera"`.
        ///
        /// On Raspberry Pi with `libcamera-still` installed, `"camera"`
        /// captures live frames via the Pi Camera Module.
        #[arg(short, long, default_value = "0")]
        source: String,

        /// Directory where captured JPEG frames will be saved.
        ///
        /// Created automatically if it does not exist.
        #[arg(short, long, default_value = "data/raw")]
        output: String,

        /// Target frame-capture rate in frames-per-second.
        ///
        /// The actual rate depends on the source speed; this is the
        /// **maximum** rate at which frames are saved (time-throttled).
        #[arg(short, long, default_value_t = 2.0)]
        rate: f32,

        /// Maximum number of frames to save before stopping.
        ///
        /// `0` = unlimited (stop when the source ends).
        #[arg(short = 'n', long, default_value_t = 0)]
        max_frames: u64,

        /// Path to the YAML config (used for camera intrinsics).
        #[arg(short, long, default_value = "configs/default.yaml")]
        config: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
//  Entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Program entry point.
///
/// Initialises logging, parses CLI arguments, and dispatches to the
/// requested subcommand handler.
fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .format_timestamp_millis()
    .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            source,
            config,
            visualize,
            ego_speed,
        } => {
            if let Err(e) = run_pipeline(&source, &config, visualize, ego_speed) {
                log::error!("Pipeline failed: {e}");
                std::process::exit(1);
            }
        }
        Commands::Collect {
            source,
            output,
            rate,
            max_frames,
            config: _config,
        } => {
            if let Err(e) = collect_data(&source, &output, rate, max_frames) {
                log::error!("Data collection failed: {e}");
                std::process::exit(1);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Run Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Executes the perception pipeline for a single video source.
///
/// # Pipeline stages
///
/// 1. **Config** — load YAML (or defaults).
/// 2. **Init** — create detector, tracker, intersection analyzer, lane-speed
///    analyzer.
/// 3. **Frame loop** — for each frame:
///    - Run YOLO detection → get [`Detection`]s.
///    - Update Deep SORT tracker → get [`Track`]s.
///    - Run intersection analysis → alerts.
///    - Run lane-speed analysis → alerts.
///    - Log alerts.
///    - (Optional) save annotated overlay frame to `./output/`.
///
/// # Parameters
/// - `source` — Video file path, image directory, `"camera"`, or device node.
/// - `config_path` — Path to YAML config (falls back to defaults).
/// - `visualize` — If `true`, rendered frames are saved to `./output/`.
/// - `ego_speed` — Ego speed in mph (for testing pre-recorded footage).
///
/// # Returns
/// - `Ok(())` on normal completion.
/// - `Err(String)` on initialisation or I/O errors.
///
/// # Panics
/// Never panics.  Errors are returned as `Err(String)`.
fn run_pipeline(
    source: &str,
    config_path: &str,
    visualize: bool,
    ego_speed: f32,
) -> Result<(), String> {
    let config = Config::load_or_default(config_path);
    log::info!("Config loaded. Model: {}", config.model.path);

    // ── Initialize components ────────────────────────────────────────
    let detector = YoloDetector::new(YoloConfig::from(&config.model))?;
    let mut tracker = MultiObjectTracker::new(
        config.tracking.max_age,
        config.tracking.n_init,
        config.tracking.max_cosine_distance,
    );
    let mut intersection_analyzer = IntersectionAnalyzer::new(&config);
    let mut lane_speed_analyzer = LaneSpeedAnalyzer::new(&config);

    // ── Open video source ────────────────────────────────────────────
    let (mut frame_iter, frame_width, frame_height) =
        open_video_source(source, config.camera.frame_width, config.camera.frame_height)?;

    log::info!(
        "Pipeline started. Source: {source}, Resolution: {frame_width}×{frame_height}, Visualize: {visualize}"
    );
    log::info!(
        "Model available: {} — detections will be empty until an ONNX model is placed at '{}'",
        detector.is_model_available(),
        config.model.path
    );

    // ── Inference loop ───────────────────────────────────────────────
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

        // Detection
        let detections = detector.detect(&frame_buffer, frame_width, frame_height)?;
        if !detections.is_empty() {
            log::debug!("Frame {frame_count}: {} detections", detections.len());
        }

        // Tracking
        let tracks = tracker.update(&detections);

        // Analysis modules
        let intersection_alerts =
            intersection_analyzer.analyze(&detections, ego_speed, dt_secs);
        let lane_alerts = lane_speed_analyzer.analyze(&tracks, ego_speed, dt_secs);

        // Dispatch alerts
        for alert in &intersection_alerts {
            match alert {
                civicsense::modules::intersection::IntersectionAlert::StopSignViolation {
                    confidence,
                    distance_to_stop_line,
                    ego_speed,
                } => {
                    log::warn!(
                        "🛑 STOP SIGN VIOLATION! conf={:.2}, dist={:.1}ft, speed={:.1}mph",
                        confidence,
                        distance_to_stop_line,
                        ego_speed
                    );
                }
                civicsense::modules::intersection::IntersectionAlert::BlockedIntersection {
                    confidence,
                    occupancy_pct,
                    distance_to_stop_line,
                    ego_speed,
                } => {
                    log::warn!(
                        "⛔ BLOCKED INTERSECTION! conf={:.2}, occupancy={:.1}%, dist={:.1}ft, speed={:.1}mph",
                        confidence,
                        occupancy_pct,
                        distance_to_stop_line,
                        ego_speed
                    );
                }
            }
        }

        for alert in &lane_alerts {
            log::warn!(
                "➡️ MERGE RIGHT REMINDER! Right lane is {:.1} mph faster (for {:.1}s)",
                alert.speed_diff_mph,
                alert.duration_secs
            );
        }

        // Visualization
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
            if let Err(e) = save_frame(&viz_frame, frame_width, frame_height, &out_path) {
                log::warn!("Failed to save visualization frame: {e}");
            }
        }

        frame_count += 1;

        // Dev limit: stop after 300 frames when not visualizing.
        if frame_count >= 300 && !visualize {
            log::info!("Processed {frame_count} frames (dev limit). Pass --visualize for output.");
            break;
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Data Collection
// ─────────────────────────────────────────────────────────────────────────────

/// Captures frames from a source and saves them as timestamped JPEGs.
///
/// Used to build a training dataset for YOLO fine-tuning.  Frames are
/// saved at a throttled rate (default 2 fps) to avoid storing near-identical
/// consecutive frames.
///
/// # Parameters
/// - `source` — Video file path, image directory, or `"camera"`.
/// - `output_dir` — Directory to write JPEG files into (created if missing).
/// - `rate` — Maximum save rate in frames-per-second (time-throttled).
/// - `max_frames` — Stop after saving this many frames (`0` = unlimited).
///
/// # Returns
/// - `Ok(())` on completion.
/// - `Err(String)` if the output directory cannot be created or the source
///   cannot be opened.
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
        open_video_source(source, 1280, 720)?;

    log::info!(
        "📸 Data collection started. Source: {source} → Output: {output_dir}/"
    );
    log::info!(
        "Resolution: {frame_width}×{frame_height}, Rate: {rate} fps, Max frames: {}",
        if max_frames == 0 {
            "unlimited".into()
        } else {
            max_frames.to_string()
        }
    );

    // Minimum interval between saves in milliseconds.
    let min_interval_ms = if rate > 0.0 {
        (1000.0 / rate) as u64
    } else {
        0
    };

    let start = Instant::now();
    let mut saved_count: u64 = 0;
    // Initialise last_save to the distant past so the first frame is always saved.
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

        // Time-based throttling: only save if enough time has passed.
        let elapsed_since_last = last_save.elapsed().as_millis() as u64;
        if elapsed_since_last >= min_interval_ms {
            let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f");
            let filename = format!("capture_{}_{:06}.jpg", timestamp, saved_count);
            let out_path = output_path.join(&filename);

            if let Err(e) = save_frame(&frame_buffer, frame_width, frame_height, &out_path) {
                log::warn!("Failed to save frame: {e}");
            } else {
                log::info!("💾 Saved: {}", out_path.display());
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
        "✅ Data collection complete. {saved_count} frames saved in {elapsed:.1?} ({effective_fps:.1} fps avg)"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Video Source Abstraction
// ─────────────────────────────────────────────────────────────────────────────

/// Type alias for a frame-iterator closure.
///
/// Returns `Some((rgb_buffer, frame_index))` for each frame, or `None`
/// when the source is exhausted.
type FrameIter = Box<dyn FnMut() -> Option<(Vec<u8>, u64)>>;

/// Opens a video/image source and returns a frame iterator.
///
/// # Supported source strings
///
/// | Pattern | Behaviour |
/// |---------|-----------|
/// | `"video.mp4"` (or .avi/.mov/…) | Extracts a single frame (thumbnail) from the file.  Full video decoding requires ffmpeg integration (TODO). |
/// | `"image.jpg"` | Single image → one frame. |
/// | `"/path/to/dir/"` | Sorted directory of images → frame per file. |
/// | `"camera"` or `"0"` | Live camera: `libcamera-still` on Raspberry Pi, instructions on macOS. |
/// | `"/dev/video0"` | V4L2 device (Linux, placeholder). |
///
/// # Parameters
/// - `source` — Source identifier (see table above).
/// - `default_width` — Fallback frame width when the source doesn't specify.
/// - `default_height` — Fallback frame height.
///
/// # Returns
/// A tuple `(frame_iter, width, height)`.
///
/// # Errors
/// Returns `Err` if the source string doesn't match any known pattern or
/// the path doesn't exist.
fn open_video_source(
    source: &str,
    default_width: u32,
    default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    let path = std::path::Path::new(source);

    // Existing file?
    if path.exists() && path.is_file() {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ["mp4", "avi", "mov", "mkv", "webm", "m4v"].contains(&ext.as_str()) {
            log::info!("Opening video file: {source}");
            return open_video_file(path, default_width, default_height);
        }
        if ext == "jpg" || ext == "jpeg" || ext == "png" {
            log::info!("Opening single image: {source}");
            return open_single_image(path, default_width, default_height);
        }
    }

    // Existing directory?
    if path.exists() && path.is_dir() {
        log::info!("Opening image directory: {source}");
        return open_image_directory(path, default_width, default_height);
    }

    // Camera mode?
    if source == "camera" || source == "0" {
        log::info!("Camera mode: {source}");
        return open_camera(source, default_width, default_height);
    }

    // V4L2 device on Linux?
    let dev_path = format!("/dev/video{}", source);
    if std::path::Path::new(&dev_path).exists() {
        log::info!("Opening V4L2 device: {dev_path}");
        return open_v4l2_device(&dev_path, default_width, default_height);
    }

    Err(format!(
        "Cannot open source '{source}'. Supported: video files, image directories, 'camera', or V4L2 device."
    ))
}

/// Loads a single image file as a one-frame source.
///
/// **Note on video files:** The `image` crate can only decode one frame
/// from a video file.  For multi-frame extraction, an ffmpeg-based
/// pipeline is needed (future work).
///
/// # Parameters
/// - `path` — Path to an image file (JPEG, PNG, etc.).
/// - `_default_width`, `_default_height` — Ignored; actual dimensions come
///   from the decoded image.
///
/// # Returns
/// A frame iterator that yields exactly one frame.
fn open_video_file(
    path: &std::path::Path,
    _default_width: u32,
    _default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    let img = image::open(path)
        .map_err(|e| format!("Failed to open image file: {e}"))?
        .into_rgb8();
    let (w, h) = img.dimensions();
    let buffer = img.into_raw();
    let buffer_clone = buffer.clone();

    let mut called = false;

    Ok((
        Box::new(move || {
            if called {
                return None;
            }
            called = true;
            Some((buffer_clone.clone(), 0))
        }),
        w,
        h,
    ))
}

/// Wrapper around [`open_video_file`] that reads a single image.
///
/// # Parameters
/// - `path` — Path to an image file.
/// - `_default_width`, `_default_height` — Ignored.
///
/// # Returns
/// A single-frame iterator.
fn open_single_image(
    path: &std::path::Path,
    _default_width: u32,
    _default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    open_video_file(path, 0, 0)
}

/// Iterates over a directory of image files, sorted alphabetically.
///
/// Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`.
///
/// # Parameters
/// - `path` — Directory containing images.
/// - `_default_width`, `_default_height` — Ignored (sized from first image).
///
/// # Returns
/// A frame iterator yielding one frame per image file.
///
/// # Errors
/// Returns `Err` if the directory cannot be read or contains no images.
fn open_image_directory(
    path: &std::path::Path,
    _default_width: u32,
    _default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    let mut entries: Vec<_> = std::fs::read_dir(path)
        .map_err(|e| format!("Cannot read directory: {e}"))?
        .filter_map(|r| r.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ["jpg", "jpeg", "png", "bmp"].contains(&ext))
                .unwrap_or(false)
        })
        .collect();

    entries.sort_by_key(|e| e.path());

    if entries.is_empty() {
        return Err("No image files found in directory".into());
    }

    let first_img = image::open(entries[0].path())
        .map_err(|e| format!("Cannot open first image: {e}"))?
        .into_rgb8();
    let (w, h) = first_img.dimensions();

    let paths: Vec<_> = entries.into_iter().map(|e| e.path()).collect();
    let mut index = 0;

    Ok((
        Box::new(move || {
            if index >= paths.len() {
                return None;
            }
            let img = image::open(&paths[index])
                .ok()?
                .into_rgb8()
                .into_raw();
            let idx = index;
            index += 1;
            Some((img, idx as u64))
        }),
        w,
        h,
    ))
}

/// Opens a camera device for live capture.
///
/// # Backend detection order
///
/// 1. `libcamera-still` — modern Raspberry Pi OS (Bookworm+).
/// 2. `raspistill` — legacy Raspberry Pi OS.
/// 3. If neither is found, logs setup instructions and returns a dummy frame.
///
/// # Parameters
/// - `_source` — The original source string (unused, kept for logging).
/// - `default_width` — Desired capture width.
/// - `default_height` — Desired capture height.
///
/// # Returns
/// A frame iterator that captures one image per call via shelling out.
/// On macOS / non-Pi systems, yields a single gray test frame.
fn open_camera(
    _source: &str,
    default_width: u32,
    default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    // Check for libcamera (modern Pi).
    let has_libcamera = std::process::Command::new("which")
        .arg("libcamera-still")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if has_libcamera {
        log::info!("📷 Raspberry Pi camera detected (libcamera)");
        return open_libcamera_camera(default_width, default_height);
    }

    // Check for raspistill (legacy Pi).
    let has_raspistill = std::process::Command::new("which")
        .arg("raspistill")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if has_raspistill {
        log::info!("📷 Raspberry Pi camera detected (raspistill)");
        return open_raspistill_camera(default_width, default_height);
    }

    // No camera backend found → give instructions.
    log::warn!(
        "No camera backend found. To collect data on Raspberry Pi:\n\
         1. Install: sudo apt install libcamera-apps\n\
         2. Run: cargo run --bin civicsense -- collect --source camera\n\
        \n\
        For now, use a video file: cargo run --bin civicsense -- collect --source video.mp4"
    );

    let buffer = vec![128u8; (default_width * default_height * 3) as usize];
    let mut called = false;
    Ok((
        Box::new(move || {
            if called {
                return None;
            }
            called = true;
            Some((buffer.clone(), 0))
        }),
        default_width,
        default_height,
    ))
}

/// Spawns `libcamera-still` for each frame, reads the captured JPEG,
/// and returns the decoded RGB buffer.
///
/// # Parameters
/// - `width` — Capture width in pixels.
/// - `height` — Capture height in pixels.
///
/// # Returns
/// A frame iterator; each call shells out to `libcamera-still`, waits for
/// capture, and decodes the result.
fn open_libcamera_camera(
    width: u32,
    height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    let tmp_dir = std::env::temp_dir().join("civicsense_capture");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| format!("Cannot create temp dir: {e}"))?;

    let mut frame_idx: u64 = 0;
    let w = width;
    let h = height;

    Ok((
        Box::new(move || {
            let capture_path = tmp_dir.join(format!("capture_{}.jpg", frame_idx));

            let status = std::process::Command::new("libcamera-still")
                .args([
                    "-o",
                    capture_path.to_str().unwrap(),
                    "--width",
                    &w.to_string(),
                    "--height",
                    &h.to_string(),
                    "--nopreview",
                    "--timeout",
                    "1",
                    "--immediate",
                ])
                .output();

            match status {
                Ok(output) if output.status.success() => {
                    match image::open(&capture_path) {
                        Ok(img) => {
                            let rgb = img.into_rgb8();
                            let buffer = rgb.into_raw();
                            let idx = frame_idx;
                            frame_idx += 1;
                            let _ = std::fs::remove_file(&capture_path);
                            Some((buffer, idx))
                        }
                        Err(e) => {
                            log::error!("Failed to decode captured image: {e}");
                            None
                        }
                    }
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    log::error!("libcamera-still failed: {stderr}");
                    None
                }
                Err(e) => {
                    log::error!("Failed to run libcamera-still: {e}");
                    None
                }
            }
        }),
        w,
        h,
    ))
}

/// Legacy Raspberry Pi camera backend using `raspistill`.
///
/// # Parameters
/// - `width` — Capture width in pixels.
/// - `height` — Capture height in pixels.
///
/// # Returns
/// A frame iterator; each call shells out to `raspistill`.
fn open_raspistill_camera(
    width: u32,
    height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    let tmp_dir = std::env::temp_dir().join("civicsense_capture");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| format!("Cannot create temp dir: {e}"))?;

    let mut frame_idx: u64 = 0;
    let w = width;
    let h = height;

    Ok((
        Box::new(move || {
            let capture_path = tmp_dir.join(format!("capture_{}.jpg", frame_idx));

            let status = std::process::Command::new("raspistill")
                .args([
                    "-o",
                    capture_path.to_str().unwrap(),
                    "-w",
                    &w.to_string(),
                    "-h",
                    &h.to_string(),
                    "-t",
                    "1",
                    "-n",
                ])
                .output();

            match status {
                Ok(output) if output.status.success() => {
                    match image::open(&capture_path) {
                        Ok(img) => {
                            let rgb = img.into_rgb8();
                            let buffer = rgb.into_raw();
                            let idx = frame_idx;
                            frame_idx += 1;
                            let _ = std::fs::remove_file(&capture_path);
                            Some((buffer, idx))
                        }
                        Err(e) => {
                            log::error!("Failed to decode captured image: {e}");
                            None
                        }
                    }
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    log::error!("raspistill failed: {stderr}");
                    None
                }
                Err(e) => {
                    log::error!("Failed to run raspistill: {e}");
                    None
                }
            }
        }),
        w,
        h,
    ))
}

/// V4L2 device capture (Linux, placeholder).
///
/// # Parameters
/// - `_dev_path` — Path to the V4L2 device (e.g. `/dev/video0`).
/// - `default_width`, `default_height` — Fallback frame dimensions.
///
/// # Returns
/// A single-frame iterator with a gray test pattern (V4L2 capture not yet
/// implemented).
fn open_v4l2_device(
    _dev_path: &str,
    default_width: u32,
    default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    log::warn!(
        "V4L2 device capture not yet implemented. \
         Use 'collect --source video.mp4' or the libcamera backend."
    );
    let buffer = vec![128u8; (default_width * default_height * 3) as usize];
    let mut called = false;
    Ok((
        Box::new(move || {
            if called {
                return None;
            }
            called = true;
            Some((buffer.clone(), 0))
        }),
        default_width,
        default_height,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
//  Frame Saving
// ─────────────────────────────────────────────────────────────────────────────

/// Saves a raw RGB8 pixel buffer as a JPEG file on disk.
///
/// # Parameters
/// - `buffer` — Flattened `(H × W × 3)` RGB8 pixel data.
/// - `width` — Image width in pixels.
/// - `height` — Image height in pixels.
/// - `path` — Destination file path (should end in `.jpg`).
///
/// # Returns
/// - `Ok(())` on success.
/// - `Err(String)` if the buffer dimensions are invalid or the file cannot
///   be written.
fn save_frame(
    buffer: &[u8],
    width: u32,
    height: u32,
    path: &std::path::Path,
) -> Result<(), String> {
    let img = image::RgbImage::from_raw(width, height, buffer.to_vec())
        .ok_or_else(|| "Failed to create image from raw buffer".to_string())?;

    img.save(path)
        .map_err(|e| format!("Failed to save image to '{}': {e}", path.display()))?;

    Ok(())
}

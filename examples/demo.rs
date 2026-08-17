//! CivicSense proof-of-concept demo on real public driving footage.
//!
//! Runs the production pipeline modules end-to-end on a directory of
//! frames extracted from a public driving-dataset video (KITTI):
//!
//! ```text
//! frames -> YOLOv8n ONNX detection -> COCO→CivicSense class mapping
//!        -> Deep SORT tracking -> intersection / lane-speed alert
//!        -> annotated JPEG frames + alerts.srt
//! ```
//!
//! The stock public YOLO models are trained on COCO-80, while the
//! CivicSense analyzers reason in the custom 7-class vocabulary
//! (`stop_sign, traffic_light, crosswalk, vehicle, truck, bus,
//! intersection_zone`). This demo maps COCO ids onto that vocabulary so
//! the real alert logic runs unmodified.
//!
//! Usage:
//! ```text
//! cargo run --example demo --release -- demo/frames demo/output \
//!     --ego-speed 25 --fps 15
//! ```

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

use civicsense::config::Config;
use civicsense::detection::yolo::{Detection, YoloConfig, YoloDetector};
use civicsense::modules::intersection::IntersectionAlert;
use civicsense::modules::intersection::IntersectionAnalyzer;
use civicsense::modules::lane_speed::{LaneSpeedAlert, LaneSpeedAnalyzer};
use civicsense::tracking::deep_sort::MultiObjectTracker;
use civicsense::utils::visualization;
use civicsense::video;

// ─────────────────────────────────────────────────────────────────────────────
//  COCO-80 vocabulary (YOLO output order)
// ─────────────────────────────────────────────────────────────────────────────

const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Map a COCO-80 class id onto the CivicSense 7-class vocabulary used by
/// the alert analyzers. Returns `None` for classes the analyzers ignore.
fn map_coco_class(id: u32) -> Option<u32> {
    match id {
        11 => Some(0), // stop_sign
        9 => Some(1),  // traffic_light
        2 => Some(3),  // vehicle (car)
        7 => Some(4),  // truck
        5 => Some(5),  // bus
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Minimal 5x7 bitmap text renderer (no graphics dependency)
// ─────────────────────────────────────────────────────────────────────────────

fn set_pixel(frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, color: (u8, u8, u8)) {
    if x >= w || y >= h {
        return;
    }
    let idx = ((y * w + x) * 3) as usize;
    if idx + 2 < frame.len() {
        frame[idx] = color.0;
        frame[idx + 1] = color.1;
        frame[idx + 2] = color.2;
    }
}

fn fill_rect(frame: &mut [u8], w: u32, h: u32, x: u32, y: u32, rw: u32, rh: u32, color: (u8, u8, u8)) {
    for yy in y..y.saturating_add(rh) {
        for xx in x..x.saturating_add(rw) {
            set_pixel(frame, xx, yy, w, h, color);
        }
    }
}

fn draw_rect(frame: &mut [u8], w: u32, h: u32, x1: u32, y1: u32, x2: u32, y2: u32, color: (u8, u8, u8), thickness: u32) {
    for t in 0..thickness {
        for x in x1..=x2 {
            set_pixel(frame, x, y1 + t, w, h, color);
            set_pixel(frame, x, y2.saturating_sub(t), w, h, color);
        }
        for y in y1..=y2 {
            set_pixel(frame, x1 + t, y, w, h, color);
            set_pixel(frame, x2.saturating_sub(t), y, w, h, color);
        }
    }
}

/// 5x7 glyph table. Bit 4 is the leftmost pixel of each row.
fn glyph(c: char) -> [u8; 7] {
    match c {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        ' ' => [0; 7],
        '.' => [0, 0, 0, 0, 0, 0b00110, 0b00110],
        ',' => [0, 0, 0, 0, 0b00110, 0b00100, 0b01000],
        ':' => [0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        '_' => [0, 0, 0, 0, 0, 0, 0b11111],
        '%' => [0b11001, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b10011],
        '!' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0, 0b00100],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '+' => [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        '?' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0, 0b00100],
        _ => [0; 7],
    }
}

/// Renders `text` at `(x, y)` with a dark background box for legibility.
fn draw_text(
    frame: &mut [u8],
    w: u32,
    h: u32,
    x: u32,
    y: u32,
    text: &str,
    color: (u8, u8, u8),
    scale: u32,
) {
    let text_w = text.chars().count() as u32 * 6 * scale;
    fill_rect(frame, w, h, x.saturating_sub(1), y.saturating_sub(1), text_w + 2, 9 * scale, (0, 0, 0));
    let mut cx = x;
    for ch in text.chars() {
        let g = glyph(ch);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..5 {
                if (bits >> (4 - col)) & 1 == 1 {
                    fill_rect(frame, w, h, cx + col * scale, y + row as u32 * scale, scale, scale, color);
                }
            }
        }
        cx += 6 * scale;
    }
}

/// Colour for a COCO class box: green vehicles, red stop signs, yellow
/// lights, cyan people, gray everything else.
fn coco_color(class_id: u32) -> (u8, u8, u8) {
    match class_id {
        0 => (0, 255, 255),
        1 | 2 | 3 | 5 | 6 | 7 => (0, 255, 0),
        9 => (255, 255, 0),
        11 => (255, 0, 0),
        _ => (180, 180, 180),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SRT helpers
// ─────────────────────────────────────────────────────────────────────────────

fn srt_time(secs: f64) -> String {
    let ms = (secs.fract() * 1000.0) as u32;
    let total = secs as u32;
    format!(
        "{:02}:{:02}:{:02},{:03}",
        total / 3600,
        (total % 3600) / 60,
        total % 60,
        ms
    )
}

// ─────────────────────────────────────────────────────────────────────────────
//  Demo driver
// ─────────────────────────────────────────────────────────────────────────────

struct Args {
    input: String,
    output: PathBuf,
    ego_speed: f32,
    fps: f32,
    model: String,
    conf: f32,
    /// Occupancy threshold (%) above which a BLOCKED INTERSECTION alert
    /// fires. Stock config default (30%) is calibrated for a narrow-FoV
    /// dashcam; the ultra-wide KITTI lens keeps vehicle occupancy far
    /// lower, so the demo defaults to 12%.
    occ_threshold: f32,
    /// Minimum ego speed (mph) for a blocked-intersection alert.
    blocked_speed: f32,
}

fn parse_args() -> Result<Args, String> {
    let mut it = env::args().skip(1);
    let mut positional = Vec::new();
    let mut flags: HashMap<String, String> = HashMap::new();
    while let Some(arg) = it.next() {
        if arg.starts_with("--") {
            let key = arg.trim_start_matches('-').to_string();
            let val = it.next().ok_or_else(|| format!("missing value for {arg}"))?;
            flags.insert(key, val);
        } else {
            positional.push(arg);
        }
    }
    if positional.len() < 2 {
        return Err("usage: demo <input-frames-dir> <output-dir> [--ego-speed MPH] [--fps N] [--model PATH] [--conf N] [--occ PCT] [--blocked-speed MPH]".into());
    }
    let num = |k: &str, d: f32| -> Result<f32, String> {
        flags.get(k).map(|v| v.parse().map_err(|_| format!("bad {k}: {v}"))).transpose().map(|v| v.unwrap_or(d))
    };
    Ok(Args {
        input: positional[0].clone(),
        output: PathBuf::from(&positional[1]),
        ego_speed: num("ego-speed", 25.0)?,
        fps: num("fps", 15.0)?,
        model: flags.get("model").cloned().unwrap_or_else(|| "weights/yolov8n.onnx".into()),
        conf: num("conf", 0.4)?,
        occ_threshold: num("occ", 12.0)?,
        blocked_speed: num("blocked-speed", 5.0)?,
    })
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    if let Err(e) = run_demo(&args) {
        eprintln!("demo failed: {e}");
        std::process::exit(1);
    }
}

fn run_demo(args: &Args) -> Result<(), String> {
    // 1. Config with the COCO model.
    let mut cfg = Config::default();
    cfg.model.path = args.model.clone();
    cfg.model.conf_threshold = args.conf;
    cfg.camera.fps = args.fps as u32;
    cfg.model.classes = COCO_CLASSES.iter().map(|s| s.to_string()).collect();
    // Demo calibration for the ultra-wide KITTI lens (see Args docs).
    cfg.intersection.blocked_occupancy_threshold = args.occ_threshold;
    cfg.intersection.blocked_intersection_speed = args.blocked_speed;

    let mut detector = YoloDetector::new(YoloConfig::from(&cfg.model))?;
    if !detector.is_model_available() {
        return Err(format!("model not found at '{}'", cfg.model.path));
    }
    log::info!("CivicSense demo: model = {}, ego speed = {:.0} mph", cfg.model.path, args.ego_speed);

    // 2. Frame source (directory of extracted frames).
    let (mut frame_iter, frame_w, frame_h) = video::open_source(&args.input, 1280, 720)?;
    log::info!("Input: {} frames at {:.0} fps, {}x{}", args.input, args.fps, frame_w, frame_h);

    std::fs::create_dir_all(&args.output).map_err(|e| format!("output dir: {e}"))?;

    // 3. Pipeline modules.
    let mut tracker = MultiObjectTracker::new(cfg.tracking.max_age, cfg.tracking.n_init, cfg.tracking.max_cosine_distance);
    let mut intersection = IntersectionAnalyzer::new(&cfg, frame_w, frame_h);
    let mut lane_speed = LaneSpeedAnalyzer::new(&cfg);

    let dt = 1.0 / args.fps;

    let mut frame_count: u64 = 0;
    let mut alert_counts = HashMap::new();
    let mut class_counts = HashMap::new();
    let mut srt = String::new();
    let mut srt_entries = 0u32;

    while let Some((buf, _idx)) = frame_iter() {
        let t = frame_count as f64 / args.fps as f64;

        // Detect on the real COCO model.
        let detections = detector.detect(&buf, frame_w, frame_h)?;
        for d in &detections {
            *class_counts.entry(COCO_CLASSES.get(d.class_id as usize).unwrap_or(&"?").to_string()).or_insert(0) += 1;
        }

        // Map onto the CivicSense vocabulary for tracking + alerts.
        let mapped: Vec<Detection> = detections
            .iter()
            .filter_map(|d| {
                map_coco_class(d.class_id).map(|c| Detection {
                    x1: d.x1,
                    y1: d.y1,
                    x2: d.x2,
                    y2: d.y2,
                    confidence: d.confidence,
                    class_id: c,
                })
            })
            .collect();

        let tracks = tracker.update(&mapped);
        let i_alerts = intersection.analyze(&mapped, args.ego_speed, dt);
        let l_alerts = lane_speed.analyze(&tracks, args.ego_speed, dt);

        // 4. Draw everything.
        let mut viz = buf.to_vec();
        for d in &detections {
            let (x1, y1) = (d.x1 as u32, d.y1 as u32);
            let (x2, y2) = (d.x2 as u32, d.y2 as u32);
            let color = coco_color(d.class_id);
            draw_rect(&mut viz, frame_w, frame_h, x1, y1, x2, y2, color, 2);
            let label = format!(
                "{} {:.2}",
                COCO_CLASSES.get(d.class_id as usize).unwrap_or(&"?").to_uppercase(),
                d.confidence
            );
            let label_y = y1.saturating_sub(10);
            draw_text(&mut viz, frame_w, frame_h, x1 + 2, label_y, &label, color, 1);
        }
        for tr in tracks.iter().filter(|t| t.is_confirmed) {
            let (tx1, ty1, _, _) = tr.bbox;
            draw_text(&mut viz, frame_w, frame_h, tx1 as u32, ty1 as u32, &format!("ID {}", tr.track_id), (255, 165, 0), 1);
        }

        // Alert banner + text.
        let mut banner_msg = String::new();
        if !i_alerts.is_empty() {
            // Pick the banner label so the strip is colour-coded correctly
            // (draw_alert_text colours "STOP"/"BLOCKED" red, "MERGE" orange).
            let critical = if i_alerts
                .iter()
                .any(|a| matches!(a, IntersectionAlert::StopSignViolation { .. }))
            {
                "STOP SIGN VIOLATION"
            } else {
                "BLOCKED INTERSECTION"
            };
            visualization::draw_alert_text(&mut viz, frame_w, frame_h, critical);
            banner_msg.push_str(
                &i_alerts
                    .iter()
                    .map(|a| alert_text(a))
                    .collect::<Vec<_>>()
                    .join(" | "),
            );
        }
        if !l_alerts.is_empty() {
            visualization::draw_alert_text(&mut viz, frame_w, frame_h, "MERGE RIGHT REMINDER");
            if !banner_msg.is_empty() {
                banner_msg.push_str(" | ");
            }
            banner_msg.push_str(&l_alerts.iter().map(alert_text_lane).collect::<Vec<_>>().join(" | "));
        }
        if !banner_msg.is_empty() {
            draw_text(&mut viz, frame_w, frame_h, 4, 6, &banner_msg, (255, 255, 255), 1);
            for a in i_alerts {
                *alert_counts.entry(alert_text(&a)).or_insert(0) += 1;
                push_srt(&mut srt, &mut srt_entries, t, &alert_text(&a));
            }
            for a in l_alerts {
                *alert_counts.entry(alert_text_lane(&a)).or_insert(0) += 1;
                push_srt(&mut srt, &mut srt_entries, t, &alert_text_lane(&a));
            }
        }

        // Status line at the bottom.
        let status = format!(
            "CIVICSENSE | KITTI | FRAME {} | {:.1}S | EGO {:.0} MPH | {} DET",
            frame_count + 1,
            t,
            args.ego_speed,
            detections.len()
        );
        draw_text(&mut viz, frame_w, frame_h, 4, frame_h - 12, &status, (0, 255, 0), 1);

        let out_path = args.output.join(format!("frame_{:06}.jpg", frame_count + 1));
        video::save_frame(&viz, frame_w, frame_h, &out_path)?;
        frame_count += 1;

        if frame_count % 50 == 0 {
            log::info!("Processed {frame_count} frames...");
        }
    }

    std::fs::write(args.output.join("alerts.srt"), &srt).map_err(|e| format!("alerts.srt: {e}"))?;

    // 5. Summary.
    log::info!("Done. {} frames -> {}", frame_count, args.output.display());
    log::info!("Detection totals: {class_counts:?}");
    log::info!("Alert totals: {alert_counts:?}");
    Ok(())
}

fn alert_text(a: &IntersectionAlert) -> String {
    match a {
        IntersectionAlert::StopSignViolation { confidence, distance_to_stop_line, ego_speed } => {
            format!("STOP SIGN VIOLATION  dist={distance_to_stop_line:.0}FT  conf={confidence:.2}  ego={ego_speed:.0}MPH")
        }
        IntersectionAlert::BlockedIntersection { confidence, occupancy_pct, distance_to_stop_line, ego_speed } => {
            format!("BLOCKED INTERSECTION  occ={occupancy_pct:.0}%  dist={distance_to_stop_line:.0}FT  conf={confidence:.2}  ego={ego_speed:.0}MPH")
        }
    }
}

fn alert_text_lane(a: &LaneSpeedAlert) -> String {
    format!("MERGE RIGHT REMINDER  right+{:.0}MPH  {:.1}S", a.speed_diff_mph, a.duration_secs)
}

/// Append a complete SRT entry showing the alert for 0.8 s.
fn push_srt(srt: &mut String, idx: &mut u32, t: f64, text: &str) {
    *idx += 1;
    let start = srt_time(t);
    let end = srt_time(t + 0.8);
    srt.push_str(&format!("{idx}\n{start} --> {end}\n[{text}]\n\n"));
}

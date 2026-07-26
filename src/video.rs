//! Video source abstraction and frame I/O.
//!
//! Provides a unified [`open_source`] function that accepts a source string
//! (file, directory, camera, or V4L2 device) and returns a frame iterator
//! along with the frame dimensions.
//!
//! Also provides [`save_frame`] for writing raw RGB buffers to JPEG files.

use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
//  FrameIter
// ─────────────────────────────────────────────────────────────────────────────

/// Frame iterator closure.
///
/// Returns `Some((rgb_buffer, frame_index))` for each frame, or `None` when
/// the source is exhausted.
pub type FrameIter = Box<dyn FnMut() -> Option<(Vec<u8>, u64)>>;

// ─────────────────────────────────────────────────────────────────────────────
//  SourceKind
// ─────────────────────────────────────────────────────────────────────────────

/// Classified source type derived from the user-supplied source string.
#[derive(Debug, Clone, PartialEq)]
enum SourceKind {
    /// A video file (mp4, avi, mov, mkv, webm, m4v).
    Video,
    /// A single image file (jpg, jpeg, png).
    Image,
    /// A directory of images.
    Directory,
    /// Live camera (libcamera-still, raspistill, or dummy fallback).
    Camera,
    /// A V4L2 video device (e.g. `/dev/video0`).
    V4l2Device(String),
}

/// Parse a source string into a [`SourceKind`].
fn classify_source(source: &str) -> SourceKind {
    let path = Path::new(source);

    if path.exists() && path.is_file() {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ["mp4", "avi", "mov", "mkv", "webm", "m4v"].contains(&ext.as_str()) {
            return SourceKind::Video;
        }
        if ext == "jpg" || ext == "jpeg" || ext == "png" {
            return SourceKind::Image;
        }
    }

    if path.exists() && path.is_dir() {
        return SourceKind::Directory;
    }

    if source == "camera" || source == "0" {
        return SourceKind::Camera;
    }

    let dev_path = format!("/dev/video{source}");
    if Path::new(&dev_path).exists() {
        return SourceKind::V4l2Device(dev_path);
    }

    SourceKind::Video // let the caller error
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Opens a video / image / camera source and returns a frame iterator.
///
/// # Supported source strings
///
/// | Pattern | Behaviour |
/// |---------|-----------|
/// | `"video.mp4"` (or .avi/.mov/…) | Extracts a single frame (thumbnail) — full video decoding requires ffmpeg |
/// | `"image.jpg"` | Single image → one frame |
/// | `"/path/to/dir/"` | Sorted directory of images → frame per file |
/// | `"camera"` or `"0"` | Live camera via libcamera-still / raspistill |
/// | `"/dev/video0"` | V4L2 device (placeholder) |
///
/// # Errors
/// Returns `Err` if the source cannot be identified or opened.
pub fn open_source(
    source: &str,
    default_width: u32,
    default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    let kind = classify_source(source);
    log::info!("Opening {kind:?} source: {source}");

    match kind {
        SourceKind::Video => open_video_file(Path::new(source), default_width, default_height),
        SourceKind::Image => open_single_image(Path::new(source)),
        SourceKind::Directory => open_image_directory(Path::new(source), default_width, default_height),
        SourceKind::Camera => open_camera(default_width, default_height),
        SourceKind::V4l2Device(dev_path) => open_v4l2_device(&dev_path, default_width, default_height),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  File / directory sources
// ─────────────────────────────────────────────────────────────────────────────

/// Opens a video or image file as a single-frame source.
///
/// **Note:** The `image` crate decodes only the first frame of a video file.
/// For multi-frame extraction, an ffmpeg-based pipeline is needed.
fn open_video_file(
    path: &Path,
    _default_width: u32,
    _default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    let img = image::open(path)
        .map_err(|e| format!("Failed to open image file: {e}"))?
        .into_rgb8();
    let (w, h) = img.dimensions();
    let buffer = img.into_raw();

    let mut called = false;
    Ok((
        Box::new(move || {
            if called {
                return None;
            }
            called = true;
            Some((buffer.clone(), 0))
        }),
        w,
        h,
    ))
}

/// Single image → one-frame iterator.
fn open_single_image(path: &Path) -> Result<(FrameIter, u32, u32), String> {
    open_video_file(path, 0, 0)
}

/// Sorted directory of images → frame-per-file iterator.
///
/// Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`.
fn open_image_directory(
    path: &Path,
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
            let img = image::open(&paths[index]).ok()?.into_rgb8().into_raw();
            let idx = index;
            index += 1;
            Some((img, idx as u64))
        }),
        w,
        h,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
//  Camera backends
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a dummy gray test frame of the given dimensions.
fn dummy_frame(width: u32, height: u32) -> Vec<u8> {
    vec![128u8; (width * height * 3) as usize]
}

/// Check whether a CLI tool is available on `$PATH`.
fn has_tool(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Capture a single frame via an external camera tool, decode it, and return
/// the raw RGB buffer.
fn capture_frame_via(tool: &str, args: &[&str], capture_path: &Path) -> Option<Vec<u8>> {
    let status = std::process::Command::new(tool)
        .args(args)
        .output();

    match status {
        Ok(output) if output.status.success() => {
            match image::open(capture_path) {
                Ok(img) => {
                    let rgb = img.into_rgb8();
                    let buffer = rgb.into_raw();
                    let _ = std::fs::remove_file(capture_path);
                    Some(buffer)
                }
                Err(e) => {
                    log::error!("Failed to decode captured image: {e}");
                    None
                }
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log::error!("{tool} failed: {stderr}");
            None
        }
        Err(e) => {
            log::error!("Failed to run {tool}: {e}");
            None
        }
    }
}

/// Live camera frame iterator.
///
/// Backend detection order:
/// 1. `libcamera-still` — modern Raspberry Pi OS (Bookworm+)
/// 2. `raspistill` — legacy Raspberry Pi OS
/// 3. If neither is found, logs setup instructions and returns a single dummy frame.
fn open_camera(
    default_width: u32,
    default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    let tmp_dir = std::env::temp_dir().join("civicsense_capture");
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| format!("Cannot create temp dir: {e}"))?;

    let (tool, tool_args): (&str, Box<dyn Fn(u32, u32, &Path) -> Vec<String>>) =
        if has_tool("libcamera-still") {
            log::info!("Raspberry Pi camera detected (libcamera)");
            ("libcamera-still", Box::new(|w, h, path| {
                vec![
                    "-o".into(), path.to_str().unwrap().into(),
                    "--width".into(), w.to_string(),
                    "--height".into(), h.to_string(),
                    "--nopreview".into(), "--timeout".into(), "1".into(),
                    "--immediate".into(),
                ]
            }))
        } else if has_tool("raspistill") {
            log::info!("Raspberry Pi camera detected (raspistill)");
            ("raspistill", Box::new(|w, h, path| {
                vec![
                    "-o".into(), path.to_str().unwrap().into(),
                    "-w".into(), w.to_string(),
                    "-h".into(), h.to_string(),
                    "-t".into(), "1".into(),
                    "-n".into(),
                ]
            }))
        } else {
            // No camera backend found.
            log::warn!(
                "No camera backend found. To collect data on Raspberry Pi:\n\
                 1. Install: sudo apt install libcamera-apps\n\
                 2. Run: cargo run --bin civicsense -- collect --source camera\n\
                 \n\
                 For now, use a video file: cargo run --bin civicsense -- collect --source video.mp4"
            );
            let buffer = dummy_frame(default_width, default_height);
            let mut called = false;
            return Ok((
                Box::new(move || {
                    if called {
                        return None;
                    }
                    called = true;
                    Some((buffer.clone(), 0))
                }),
                default_width,
                default_height,
            ));
        };

    let mut frame_idx: u64 = 0;
    let w = default_width;
    let h = default_height;
    let tool_owned = tool.to_string();

    Ok((
        Box::new(move || {
            let capture_path = tmp_dir.join(format!("capture_{frame_idx}.jpg"));
            let args = tool_args(w, h, &capture_path);
            let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

            if let Some(buffer) = capture_frame_via(&tool_owned, &args_refs, &capture_path) {
                let idx = frame_idx;
                frame_idx += 1;
                Some((buffer, idx))
            } else {
                None
            }
        }),
        w,
        h,
    ))
}

/// V4L2 device capture (Linux, placeholder).
///
/// Returns a single-frame iterator with a gray test pattern. V4L2 capture is
/// not yet implemented.
fn open_v4l2_device(
    _dev_path: &str,
    default_width: u32,
    default_height: u32,
) -> Result<(FrameIter, u32, u32), String> {
    log::warn!(
        "V4L2 device capture not yet implemented. \
         Use a video file or the libcamera backend."
    );
    let buffer = dummy_frame(default_width, default_height);
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
//  Frame saving
// ─────────────────────────────────────────────────────────────────────────────

/// Saves a raw RGB8 pixel buffer as a JPEG file.
///
/// # Errors
/// Returns `Err` if the buffer dimensions are invalid or the file cannot be
/// written.
pub fn save_frame(
    buffer: &[u8],
    width: u32,
    height: u32,
    path: &Path,
) -> Result<(), String> {
    let img = image::RgbImage::from_raw(width, height, buffer.to_vec())
        .ok_or_else(|| "Failed to create image from raw buffer".to_string())?;

    img.save(path)
        .map_err(|e| format!("Failed to save image to '{}': {e}", path.display()))?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_video() {
        let kind = classify_source("video.mp4");
        assert_eq!(kind, SourceKind::Video);
    }

    #[test]
    fn test_classify_image() {
        let kind = classify_source("photo.jpg");
        assert_eq!(kind, SourceKind::Video); // doesn't exist, falls to Video (will error)
    }

    #[test]
    fn test_classify_camera() {
        assert_eq!(classify_source("camera"), SourceKind::Camera);
        assert_eq!(classify_source("0"), SourceKind::Camera);
    }
}

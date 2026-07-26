//! # 🔗 Deep SORT Multi-Object Tracker
//!
//! Assigns and maintains persistent IDs for detected objects across frames
//! using a Kalman filter + greedy IoU association.
//!
//! ## Design
//!
//! - **Kalman filter**: 8-dimensional constant-velocity state
//!   `(cx, cy, w, h, vx, vy, vw, vh)` with scalar-gain approximation for
//!   the update step.
//! - **Association**: IoU-based greedy matching (Hungarian algorithm and
//!   appearance-based cosine gating are future work).
//! - **Track lifecycle**: tracks are *tentative* until `n_init` successful
//!   matches promote them to *confirmed*.  Unmatched tracks age out after
//!   `max_age` consecutive misses.

use crate::detection::yolo::Detection;
use crate::utils::geometry::compute_iou;

// ─────────────────────────────────────────────────────────────────────────────
//  Kalman Filter Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Process noise covariance added to the state diagonal each predict step.
///
/// Higher values make the filter trust the motion model less and rely more
/// on new measurements.  Default: `0.01`.
const Q_VAR: f32 = 0.01;

/// Measurement noise covariance.
///
/// Added to the innovation covariance during the update step.  Higher values
/// indicate noisier detections.  Default: `0.1`.
const R_VAR: f32 = 0.1;

/// Initial state covariance for the four position terms.
///
/// The velocity terms start at `P_INIT × 100` to reflect greater uncertainty.
/// Default: `10.0`.
const P_INIT: f32 = 10.0;

// ─────────────────────────────────────────────────────────────────────────────
//  KalmanFilter (private)
// ─────────────────────────────────────────────────────────────────────────────

/// An 8-dimensional constant-velocity Kalman filter.
///
/// **State vector** (order):  
/// `[cx, cy, w, h, vx, vy, vw, vh]`
///
/// where `(cx, cy)` is the bounding-box centre, `(w, h)` its width/height,
/// and `(vx, vy, vw, vh)` the corresponding velocities.
///
/// **Measurement vector** (order): `[cx, cy, w, h]`
///
/// The update step uses a **scalar-gain approximation** for simplicity:
/// each state dimension is corrected independently by
/// `gain = P_ii / (P_ii + R)` instead of a full matrix inversion.
#[derive(Debug, Clone)]
struct KalmanFilter {
    /// State mean: 8-element vector `[cx, cy, w, h, vx, vy, vw, vh]`.
    mean: [f32; 8],

    /// State covariance: 8×8 matrix stored **flattened row-major** (64 elements).
    ///
    /// Only the diagonal is actively maintained in this simplified filter.
    cov: [f32; 64],
}

impl KalmanFilter {
    /// Initialises the filter from a bounding-box measurement.
    ///
    /// # Parameters
    /// - `x1` — Left edge of the bounding box (pixels).
    /// - `y1` — Top edge of the bounding box (pixels).
    /// - `x2` — Right edge of the bounding box (pixels).
    /// - `y2` — Bottom edge of the bounding box (pixels).
    ///
    /// Velocity components are initialised to `0.0` with high covariance.
    ///
    /// # Returns
    /// A new `KalmanFilter` whose state is centred on the given box.
    fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        let cx = (x1 + x2) / 2.0;
        let cy = (y1 + y2) / 2.0;
        let w = (x2 - x1).abs();
        let h = (y2 - y1).abs();
        let mean = [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0];
        // Initial covariance: high uncertainty for velocity terms.
        let mut cov = [0.0f32; 64];
        for i in 0..4 {
            cov[i * 9] = P_INIT; // diagonal element cov[i][i]
        }
        for i in 4..8 {
            cov[i * 9] = P_INIT * 100.0; // higher uncertainty on velocity
        }
        Self { mean, cov }
    }

    /// Advances the state by one time step (predict).
    ///
    /// Adds velocity to position and increases covariance with process noise.
    /// Note: dt is implicitly `1.0` (one frame); a full implementation
    /// would scale velocity by the actual inter-frame delta.
    fn predict(&mut self) {
        // x += velocity (in place)
        self.mean[0] += self.mean[4]; // cx += vx
        self.mean[1] += self.mean[5]; // cy += vy
        self.mean[2] += self.mean[6]; // w  += vw
        self.mean[3] += self.mean[7]; // h  += vh

        // P += Q  (add process noise to diagonal)
        for i in 0..8 {
            self.cov[i * 9] += Q_VAR;
        }
    }

    /// Corrects the state with a new measurement (update).
    ///
    /// # Parameters
    /// - `x1`, `y1`, `x2`, `y2` — The new bounding box measurement (pixels).
    ///
    /// The innovation is computed as `z - H·x` where `H = [I₄ | 0₄]`.
    /// A scalar-gain approximation is used instead of a full Kalman gain
    /// calculation: `gain_i = P_ii / (P_ii + R)`.
    fn update(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        let cx = (x1 + x2) / 2.0;
        let cy = (y1 + y2) / 2.0;
        let w = (x2 - x1).abs();
        let h = (y2 - y1).abs();
        let z = [cx, cy, w, h];

        // Innovation: y = z - H * x  (H extracts first 4 elements of state)
        let y0 = z[0] - self.mean[0];
        let y1 = z[1] - self.mean[1];
        let y2 = z[2] - self.mean[2];
        let y3 = z[3] - self.mean[3];

        // Scalar-gain approximation
        for i in 0..4 {
            let p = self.cov[i * 9];
            let gain = p / (p + R_VAR);
            self.mean[i] += gain * [y0, y1, y2, y3][i];
            self.cov[i * 9] *= 1.0 - gain;
        }
    }

    /// Returns the predicted bounding box in `(x1, y1, x2, y2)` format.
    ///
    /// # Returns
    /// A tuple `(left, top, right, bottom)` derived from the current
    /// state-vector centre and dimensions.
    fn bbox(&self) -> (f32, f32, f32, f32) {
        let cx = self.mean[0];
        let cy = self.mean[1];
        let w = self.mean[2];
        let h = self.mean[3];
        (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Track (public)
// ─────────────────────────────────────────────────────────────────────────────

/// A single tracked object with a persistent ID across frames.
///
/// Each `Track` wraps a [`KalmanFilter`] for motion prediction and exposes
/// the predicted bounding box, age, and confirmation status.
#[derive(Debug, Clone)]
pub struct Track {
    /// Globally unique (monotonically increasing) identifier.
    ///
    /// Assigned by [`MultiObjectTracker`] at track birth.
    pub track_id: u64,

    /// Predicted bounding box `(x1, y1, x2, y2)` in absolute pixel coordinates.
    pub bbox: (f32, f32, f32, f32),

    /// Number of frames since this track was first created.
    pub age: u32,

    /// Whether this track has accumulated enough hits to be considered stable.
    ///
    /// A track becomes confirmed once its hit count ≥ `n_init` (configured
    /// in [`MultiObjectTracker`]).
    pub is_confirmed: bool,

    /// Internal Kalman filter state (position, velocity, covariance).
    kalman: KalmanFilter,

    /// Number of consecutive frames without a matching detection.
    time_since_update: u32,

    /// Total number of successful detection-to-track matches.
    hits: u32,
}

impl Track {
    /// Creates a new track from an initial detection.
    ///
    /// # Parameters
    /// - `track_id` — Unique ID assigned by the parent tracker.
    /// - `detection` — The first detection that seeds this track.
    ///
    /// # Returns
    /// A `Track` initialised with the detection's bounding box and the
    /// Kalman filter centred on that box.  The track starts as
    /// **unconfirmed** with `hits = 1`.
    pub fn new(track_id: u64, detection: &Detection) -> Self {
        let kalman = KalmanFilter::new(detection.x1, detection.y1, detection.x2, detection.y2);
        Self {
            track_id,
            bbox: (detection.x1, detection.y1, detection.x2, detection.y2),
            age: 0,
            is_confirmed: false,
            kalman,
            time_since_update: 0,
            hits: 1,
        }
    }

    /// Performs the Kalman **predict** step for one frame.
    ///
    /// Advances the state, updates the predicted bounding box, increments
    /// `age` and `time_since_update`.
    pub fn predict(&mut self) {
        self.kalman.predict();
        self.bbox = self.kalman.bbox();
        self.age += 1;
        self.time_since_update += 1;
    }

    /// Performs the Kalman **update** step with a matching detection.
    ///
    /// # Parameters
    /// - `detection` — The matched detection from the current frame.
    ///
    /// Resets `time_since_update` to `0` and increments `hits`.
    pub fn update(&mut self, detection: &Detection) {
        self.kalman
            .update(detection.x1, detection.y1, detection.x2, detection.y2);
        self.bbox = self.kalman.bbox();
        self.time_since_update = 0;
        self.hits += 1;
    }

    /// Returns the number of consecutive frames without a matching detection.
    ///
    /// # Returns
    /// Frame count since the last successful `update()` call.
    pub fn time_since_update(&self) -> u32 {
        self.time_since_update
    }

    /// Returns the total number of successful detection matches.
    ///
    /// # Returns
    /// Total hit count; used by the tracker to determine confirmation status.
    pub fn hits(&self) -> u32 {
        self.hits
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  MultiObjectTracker (public)
// ─────────────────────────────────────────────────────────────────────────────

/// Orchestrates multi-object tracking across frames using a Deep SORT-like
/// predict-match-update cycle.
///
/// ## Lifecycle
///
/// 1. **Predict** — every active track advances its Kalman state.
/// 2. **Match** — detections are paired with tracks via greedy IoU matching
///    (IoU > 0.3 gate).  Matched tracks are updated.
/// 3. **Birth** — unmatched detections spawn new tentative tracks.
/// 4. **Confirm** — tracks with ≥ `n_init` hits are marked `is_confirmed`.
/// 5. **Death** — tracks whose `time_since_update > max_age` are removed.
pub struct MultiObjectTracker {
    /// All currently active tracks (both confirmed and tentative).
    tracks: Vec<Track>,

    /// Monotonically increasing counter for assigning new track IDs.
    next_id: u64,

    /// Maximum number of unmatched frames before a track is killed.
    max_age: u32,

    /// Minimum hits required to mark a track as confirmed.
    n_init: u32,

    /// Reserved for future appearance-based cosine-distance gating.
    _max_cosine_distance: f32,
}

impl MultiObjectTracker {
    /// Creates a new empty tracker.
    ///
    /// # Parameters
    /// - `max_age` — Tracks unmatched for this many consecutive frames are
    ///   removed.  Recommended: `30` (~1 s at 30 fps).
    /// - `n_init` — Number of matched frames before a track is promoted to
    ///   confirmed.  Recommended: `3`.
    /// - `max_cosine_distance` — _(reserved)_ Future appearance-gating
    ///   threshold.  Pass `0.2` for now.
    ///
    /// # Returns
    /// A new `MultiObjectTracker` with zero active tracks.
    pub fn new(max_age: u32, n_init: u32, max_cosine_distance: f32) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_age,
            n_init,
            _max_cosine_distance: max_cosine_distance,
        }
    }

    /// Processes one frame of detections and returns the updated track list.
    ///
    /// Call this once per video frame, passing the detections produced by
    /// the YOLO detector.
    ///
    /// # Parameters
    /// - `detections` — All detections from the current frame.  Pass an
    ///   empty slice `&[]` when nothing is detected.
    ///
    /// # Returns
    /// A clone of all active `Track` objects (both confirmed and tentative)
    /// after the predict-match-update-birth-death cycle.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        self.predict_all();
        let unmatched_det = self.match_and_update(detections);
        self.birth_new_tracks(detections, &unmatched_det);
        self.confirm_tracks();
        self.remove_stale_tracks();
        self.tracks.clone()
    }

    /// Advance Kalman state for every active track.
    fn predict_all(&mut self) {
        for track in &mut self.tracks {
            track.predict();
        }
    }

    /// Greedy IoU matching between predicted tracks and detections.
    ///
    /// Matched tracks are updated in place. Returns indices of detections
    /// that did not match any track.
    fn match_and_update(&mut self, detections: &[Detection]) -> Vec<usize> {
        if self.tracks.is_empty() || detections.is_empty() {
            return (0..detections.len()).collect();
        }

        let matches = self.build_match_candidates(detections);
        self.apply_matches(detections, &matches)
    }

    /// Build IoU-gated candidate matches between all tracks and detections.
    fn build_match_candidates(&self, detections: &[Detection]) -> Vec<(usize, usize, f32)> {
        let mut candidates = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            for (di, det) in detections.iter().enumerate() {
                let iou_val = compute_iou(track.bbox, (det.x1, det.y1, det.x2, det.y2));
                if iou_val > 0.3 {
                    candidates.push((ti, di, iou_val));
                }
            }
        }
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    /// Greedily assign detections to tracks from candidate list.
    /// Returns indices of unmatched detections.
    fn apply_matches(&mut self, detections: &[Detection], matches: &[(usize, usize, f32)]) -> Vec<usize> {
        let mut used_trk = vec![false; self.tracks.len()];
        let mut used_det = vec![false; detections.len()];

        for &(ti, di, _) in matches {
            if !used_trk[ti] && !used_det[di] {
                self.tracks[ti].update(&detections[di]);
                used_trk[ti] = true;
                used_det[di] = true;
            }
        }

        detections
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_det[*i])
            .map(|(i, _)| i)
            .collect()
    }

    /// Spawn new tentative tracks for unmatched detections.
    fn birth_new_tracks(&mut self, detections: &[Detection], unmatched: &[usize]) {
        for &di in unmatched {
            let track = Track::new(self.next_id, &detections[di]);
            self.tracks.push(track);
            self.next_id += 1;
        }
    }

    /// Promote tracks with enough hits to confirmed status.
    fn confirm_tracks(&mut self) {
        for track in &mut self.tracks {
            if track.hits() >= self.n_init {
                track.is_confirmed = true;
            }
        }
    }

    /// Remove tracks that have been unmatched for too long.
    fn remove_stale_tracks(&mut self) {
        self.tracks
            .retain(|t| t.time_since_update() <= self.max_age);
    }

    /// Returns a reference to all active tracks.
    ///
    /// # Returns
    /// A slice of `Track` objects currently managed by the tracker
    /// (read-only).
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Returns the number of currently active tracks.
    ///
    /// # Returns
    /// The count of both confirmed and tentative tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a quick Detection for tests.
    fn make_det(x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection {
            x1,
            y1,
            x2,
            y2,
            confidence: 0.9,
            class_id: 0,
        }
    }

    /// A single detection in the first frame should produce one track with
    /// ID = 1.
    #[test]
    fn test_tracker_creates_tracks() {
        let mut tracker = MultiObjectTracker::new(30, 3, 0.2);
        let dets = vec![make_det(10.0, 10.0, 50.0, 50.0)];
        let tracks = tracker.update(&dets);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, 1);
    }

    /// The same object in consecutive frames should keep the same track ID.
    #[test]
    fn test_tracker_matches_over_frames() {
        let mut tracker = MultiObjectTracker::new(30, 3, 0.2);
        let dets = vec![make_det(10.0, 10.0, 50.0, 50.0)];
        let tracks1 = tracker.update(&dets);
        assert_eq!(tracks1.len(), 1);

        // Slightly shifted box in the next frame (IoU should still be high).
        let dets2 = vec![make_det(12.0, 12.0, 52.0, 52.0)];
        let tracks2 = tracker.update(&dets2);
        assert_eq!(tracks2.len(), 1);
        assert_eq!(tracks2[0].track_id, 1);
    }

    /// Tracks should be removed after exceeding `max_age` unmatched frames.
    #[test]
    fn test_tracker_birth_and_death() {
        let mut tracker = MultiObjectTracker::new(1, 1, 0.2);
        let dets = vec![make_det(10.0, 10.0, 50.0, 50.0)];
        let _ = tracker.update(&dets);

        // No detections in the next frame → track still alive (age = 1).
        let _ = tracker.update(&[]);
        assert_eq!(tracker.track_count(), 1);

        // Another frame with no detections → track exceeds max_age = 1 and dies.
        let _ = tracker.update(&[]);
        assert_eq!(tracker.track_count(), 0);
    }

    /// Identical boxes should produce an IoU of exactly 1.0.
    #[test]
    fn test_iou_same_box() {
        let bbox = (10.0, 10.0, 50.0, 50.0);
        let result = compute_iou(bbox, bbox);
        assert!((result - 1.0).abs() < 1e-6);
    }

    /// Non-overlapping boxes should produce an IoU of exactly 0.0.
    #[test]
    fn test_iou_no_overlap() {
        let result = compute_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0));
        assert!((result - 0.0).abs() < 1e-6);
    }
}

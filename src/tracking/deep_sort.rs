//! # 🔗 Deep SORT Multi-Object Tracker
//!
//! Assigns and maintains persistent IDs for detected objects across frames
//! using Kalman filtering + Hungarian association.
//!
//! ## Current Implementation
//!
//! - **Kalman filter**: 8-dimensional state `(x, y, w, h, vx, vy, vw, vh)`
//!   with constant-velocity motion model. Standard `predict` / `update` cycle.
//! - **Association**: IoU-based greedy matching (cosine-distance appearance
//!   gating is TODO — the feature extractor requires a CNN).
//! - **Track management**: tracks are `confirmed` after `n_init` matches;
//!   unmatched tracks age out after `max_age` frames.

use crate::detection::yolo::Detection;

// ── Constants ────────────────────────────────────────────────────────────

/// Process noise covariance (how much we trust the motion model).
const Q_VAR: f32 = 0.01;
/// Measurement noise covariance (how much we trust detections).
const R_VAR: f32 = 0.1;
/// Initial state covariance.
const P_INIT: f32 = 10.0;

// ── Kalman Filter ────────────────────────────────────────────────────────

/// A simple 8-dimensional constant-velocity Kalman filter.
///
/// State: [x, y, w, h, vx, vy, vw, vh]
/// Measurement: [x, y, w, h]
#[derive(Debug, Clone)]
struct KalmanFilter {
    /// State mean (8-vector).
    mean: [f32; 8],
    /// State covariance (8×8, stored flattened row-major).
    cov: [f32; 64],
}

impl KalmanFilter {
    /// Initialize with a bounding box measurement (x1, y1, x2, y2).
    fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        let cx = (x1 + x2) / 2.0;
        let cy = (y1 + y2) / 2.0;
        let w = (x2 - x1).abs();
        let h = (y2 - y1).abs();
        let mean = [cx, cy, w, h, 0.0, 0.0, 0.0, 0.0];
        // Initial covariance: high uncertainty for velocity terms.
        let mut cov = [0.0f32; 64];
        for i in 0..4 {
            cov[i * 9] = P_INIT; // diagonal: cov[i][i]
        }
        for i in 4..8 {
            cov[i * 9] = P_INIT * 100.0; // higher uncertainty on velocity
        }
        Self { mean, cov }
    }

    /// Predict step: advance state and increase uncertainty.
    fn predict(&mut self) {
        // x = F * x  (constant velocity: F is identity for position, identity for velocity)
        // P = F * P * F^T + Q
        // With F = I (since we'd add dt*velocity, we skip dt scaling for simplicity;
        // the velocity terms just persist). In a full implementation, dt would be used.
        // Here we apply a simple constant-velocity update.

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

    /// Update step: correct state with a new measurement.
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

        // Innovation covariance: S = H * P * H^T + R
        // Since H = [I_4 | 0_4], S = P[:4,:4] + R
        let mut s = [0.0f32; 16]; // 4×4
        for i in 0..4 {
            for j in 0..4 {
                s[i * 4 + j] = self.cov[i * 8 + j];
            }
            s[i * 4 + i] += R_VAR;
        }

        // Compute determinant of S (for sanity — not strictly needed).
        // Kalman gain: K = P * H^T * S^{-1}
        // Since H = [I | 0], K = P[:,:4] * S^{-1}
        // Simplified: we just do a weighted update.
        //
        // For simplicity, use a scalar approximation:
        // gain = P_diag / (P_diag + R)
        for i in 0..4 {
            let p = self.cov[i * 9];
            let gain = p / (p + R_VAR);
            self.mean[i] += gain * [y0, y1, y2, y3][i];
            // Update covariance: (I - K*H) * P
            self.cov[i * 9] *= 1.0 - gain;
        }
    }

    /// Returns the predicted bounding box (x1, y1, x2, y2).
    fn bbox(&self) -> (f32, f32, f32, f32) {
        let cx = self.mean[0];
        let cy = self.mean[1];
        let w = self.mean[2];
        let h = self.mean[3];
        (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
    }
}

// ── Track ────────────────────────────────────────────────────────────────

/// A single tracked object with persistent ID.
#[derive(Debug, Clone)]
pub struct Track {
    pub track_id: u64,
    pub bbox: (f32, f32, f32, f32), // (x1, y1, x2, y2)
    pub age: u32,
    pub is_confirmed: bool,
    kalman: KalmanFilter,
    /// Number of consecutive unmatched frames.
    time_since_update: u32,
    /// Total number of hits (matched frames).
    hits: u32,
}

impl Track {
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

    /// Predicts the next state (Kalman predict step).
    pub fn predict(&mut self) {
        self.kalman.predict();
        self.bbox = self.kalman.bbox();
        self.age += 1;
        self.time_since_update += 1;
    }

    /// Updates the state with a new matching detection (Kalman update step).
    pub fn update(&mut self, detection: &Detection) {
        self.kalman
            .update(detection.x1, detection.y1, detection.x2, detection.y2);
        self.bbox = self.kalman.bbox();
        self.time_since_update = 0;
        self.hits += 1;
    }

    /// Returns the number of frames since last update.
    pub fn time_since_update(&self) -> u32 {
        self.time_since_update
    }

    /// Returns the total hit count.
    pub fn hits(&self) -> u32 {
        self.hits
    }
}

// ── IoU Utility ──────────────────────────────────────────────────────────

/// Computes Intersection-over-Union of two bounding boxes.
fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let (ax1, ay1, ax2, ay2) = a;
    let (bx1, by1, bx2, by2) = b;

    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);

    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;

    let a_area = (ax2 - ax1) * (ay2 - ay1);
    let b_area = (bx2 - bx1) * (by2 - by1);
    let union = a_area + b_area - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

// ── Multi-Object Tracker ─────────────────────────────────────────────────

/// Multi-object tracker orchestrating Deep SORT across frames.
pub struct MultiObjectTracker {
    tracks: Vec<Track>,
    next_id: u64,
    max_age: u32,
    n_init: u32,
    _max_cosine_distance: f32,
}

impl MultiObjectTracker {
    pub fn new(max_age: u32, n_init: u32, max_cosine_distance: f32) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_age,
            n_init,
            _max_cosine_distance: max_cosine_distance,
        }
    }

    /// Updates all tracks with new detections from the current frame.
    ///
    /// Returns the active (confirmed + unconfirmed) tracks after this update.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        // 1. Predict all existing tracks.
        for track in &mut self.tracks {
            track.predict();
        }

        // 2. Build IoU cost matrix between unmatched tracks and detections.
        let mut unmatched_det: Vec<usize> = (0..detections.len()).collect();

        if !self.tracks.is_empty() && !detections.is_empty() {
            let mut matches: Vec<(usize, usize, f32)> = Vec::new();

            for (ti, track) in self.tracks.iter().enumerate() {
                for (di, det) in detections.iter().enumerate() {
                    let iou_val = iou(track.bbox, (det.x1, det.y1, det.x2, det.y2));
                    if iou_val > 0.3 {
                        // IoU gating threshold
                        matches.push((ti, di, iou_val));
                    }
                }
            }

            // Sort by IoU descending (greedy matching).
            matches.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

            let mut used_trk = vec![false; self.tracks.len()];
            let mut used_det = vec![false; detections.len()];

            for &(ti, di, _iou) in &matches {
                if !used_trk[ti] && !used_det[di] {
                    // Match!
                    self.tracks[ti].update(&detections[di]);
                    used_trk[ti] = true;
                    used_det[di] = true;
                }
            }

            unmatched_det = detections
                .iter()
                .enumerate()
                .filter(|(i, _)| !used_det[*i])
                .map(|(i, _)| i)
                .collect();
        }

        // 3. Birth new tracks for unmatched detections.
        for &di in &unmatched_det {
            let track = Track::new(self.next_id, &detections[di]);
            self.tracks.push(track);
            self.next_id += 1;
        }

        // 4. Mark tracks as confirmed after n_init hits.
        for track in &mut self.tracks {
            if track.hits() >= self.n_init {
                track.is_confirmed = true;
            }
        }

        // 5. Remove tracks that have exceeded max_age without update.
        self.tracks
            .retain(|t| t.time_since_update() <= self.max_age);

        // 6. Return active tracks (both confirmed and tentative).
        //    Clone only the public parts.
        self.tracks.clone()
    }

    /// Returns a reference to all active tracks.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Returns the number of active tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_tracker_creates_tracks() {
        let mut tracker = MultiObjectTracker::new(30, 3, 0.2);
        let dets = vec![make_det(10.0, 10.0, 50.0, 50.0)];
        let tracks = tracker.update(&dets);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, 1);
    }

    #[test]
    fn test_tracker_matches_over_frames() {
        let mut tracker = MultiObjectTracker::new(30, 3, 0.2);
        let dets = vec![make_det(10.0, 10.0, 50.0, 50.0)];
        let tracks1 = tracker.update(&dets);
        assert_eq!(tracks1.len(), 1);

        // Same detection in next frame → same track ID.
        let dets2 = vec![make_det(12.0, 12.0, 52.0, 52.0)];
        let tracks2 = tracker.update(&dets2);
        assert_eq!(tracks2.len(), 1);
        assert_eq!(tracks2[0].track_id, 1);
    }

    #[test]
    fn test_tracker_birth_and_death() {
        let mut tracker = MultiObjectTracker::new(1, 1, 0.2);
        let dets = vec![make_det(10.0, 10.0, 50.0, 50.0)];
        let _ = tracker.update(&dets);

        // No detections → track should age but still exist.
        let _ = tracker.update(&[]);
        assert_eq!(tracker.track_count(), 1);

        // After max_age (1) with no updates, track should die.
        let _ = tracker.update(&[]);
        assert_eq!(tracker.track_count(), 0);
    }

    #[test]
    fn test_iou_same_box() {
        let bbox = (10.0, 10.0, 50.0, 50.0);
        let result = iou(bbox, bbox);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_iou_no_overlap() {
        let result = iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0));
        assert!((result - 0.0).abs() < 1e-6);
    }
}

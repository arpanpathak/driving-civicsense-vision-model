//! # 🔗 Deep SORT Multi-Object Tracker
//!
//! Assigns and maintains persistent IDs for detected vehicles across frames
//! using Kalman filtering + Hungarian association with appearance features.
//!
//! ## TODO
//!
//! - [ ] Implement Kalman filter (8-dimensional state: x, y, w, h, vx, vy, vw, vh)
//! - [ ] Implement Hungarian algorithm for detection-track assignment
//! - [ ] Add appearance feature extractor (simple CNN embedding)
//! - [ ] Implement track management (birth, death, age, confirmed)

#![allow(unused_variables, dead_code)]

use crate::detection::yolo::Detection;

/// A single tracked object with persistent ID.
#[derive(Debug, Clone)]
pub struct Track {
    pub track_id: u64,
    pub bbox: (f32, f32, f32, f32), // (x1, y1, x2, y2)
    pub age: u32,
    pub is_confirmed: bool,
    // TODO: Kalman filter state, appearance feature vector
}

impl Track {
    pub fn new(track_id: u64, detection: &Detection) -> Self {
        todo!("Initialize track with Kalman state from detection");
    }

    /// Predicts the next state (Kalman predict step).
    pub fn predict(&mut self) {
        todo!("Kalman filter predict step");
    }

    /// Updates the state with a new matching detection (Kalman update step).
    pub fn update(&mut self, detection: &Detection) {
        todo!("Kalman filter update step with new measurement");
    }
}

/// Multi-object tracker orchestrating Deep SORT across frames.
pub struct MultiObjectTracker {
    tracks: Vec<Track>,
    next_id: u64,
    // TODO: config (max_age, n_init, max_cosine_distance)
}

impl MultiObjectTracker {
    pub fn new(max_age: u32, n_init: u32, max_cosine_distance: f32) -> Self {
        todo!("Initialize tracker with config");
    }

    /// Updates all tracks with new detections from the current frame.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        todo!("Run association → update → manage tracks");
    }
}

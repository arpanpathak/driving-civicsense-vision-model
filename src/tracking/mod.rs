//! 🔗 Multi-Object Tracking Module
//!
//! Assigns persistent Track IDs to detected vehicles for relative speed
//! estimation and lane assignment. Wraps a Deep SORT / BoT-SORT algorithm.
//!
//! ## 📋 TODO
//!
//! - [ ] Implement Deep SORT with Kalman filter + Hungarian algorithm
//! - [ ] Add feature extractor (simple CNN) for Re-ID embeddings
//! - [ ] Handle track birth/death with configurable max age
//! - [ ] Add occlusion recovery logic

pub mod deep_sort;

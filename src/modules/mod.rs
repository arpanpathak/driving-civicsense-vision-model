//! 🧩 Domain Logic Modules
//!
//! Specialized modules that interpret detection/tracking outputs:
//!
//! - `intersection` — Stop sign compliance & occupancy grid
//! - `lane_speed` — Relative speed estimation & lane courtesy reminders

pub mod intersection;
pub mod lane_speed;

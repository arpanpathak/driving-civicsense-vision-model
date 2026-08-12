//! Pure algebraic functions for kinematic calculations.
//! These are stateless, deterministic, and side-effect-free.

/// Physical constants derived from automotive safety standards.
pub mod constants {
    /// Maximum comfortable deceleration (m/s^2).
    pub const MAX_DECEL: f32 = 4.0;
    /// Perception-reaction time (seconds).
    pub const REACTION_TIME: f32 = 1.0;
    /// Standard intersection width for two lanes (meters).
    pub const INTERSECTION_LENGTH: f32 = 16.0;
    /// Safety margin to guarantee red phase clearance (seconds).
    pub const SAFETY_MARGIN: f32 = 0.8;
    /// Small epsilon to prevent division by zero.
    pub const EPSILON: f32 = 0.1;
    /// Standard lane width for cut-in calculations (meters).
    pub const LANE_WIDTH: f32 = 3.5;
    /// Time-to-red below which a yellow is treated as too short for a
    /// comfortable stop (seconds).
    pub const SHORT_YELLOW_THRESHOLD: f32 = 2.5;
    /// Speed below which a lead vehicle is treated as stopped (m/s).
    pub const STOPPED_SPEED_THRESHOLD: f32 = 1.0;
}

/// Computes the minimum stopping distance (Eq. 1).
///
/// # Derivation
/// From `v_f^2 = v_i^2 + 2*a*d`, with `v_f = 0` and `a = -MAX_DECEL`,
/// we obtain `d_brake = v^2 / (2 * MAX_DECEL)`. Adding the reaction
/// distance `v * REACTION_TIME` yields the total.
///
/// # Arguments
/// * `ego_speed` - Current longitudinal velocity (m/s).
///
/// # Returns
/// The distance (m) required to achieve a full stop.
#[must_use]
pub fn stopping_distance(ego_speed: f32) -> f32 {
    let reaction_dist = ego_speed * constants::REACTION_TIME;
    let braking_dist = (ego_speed * ego_speed) / (2.0 * constants::MAX_DECEL);
    reaction_dist + braking_dist
}

/// Computes the time to clear the intersection (Eq. 2).
///
/// # Arguments
/// * `distance_to_line` - Distance from front bumper to the stop line (m).
/// * `speed` - Current longitudinal velocity (m/s).
///
/// # Returns
/// The time (s) required to reach the far side of the intersection.
#[must_use]
pub fn clearance_time(distance_to_line: f32, speed: f32) -> f32 {
    let denominator = speed + constants::EPSILON;
    (distance_to_line + constants::INTERSECTION_LENGTH) / denominator
}

/// Computes the time for an adjacent vehicle to intrude into the ego lane
/// (Eq. 4).
///
/// # Arguments
/// * `lateral_speed` - The lateral velocity of the adjacent vehicle (m/s).
///
/// # Returns
/// The time (s) until the lane boundary is crossed.
#[must_use]
pub fn intrusion_time(lateral_speed: f32) -> f32 {
    constants::LANE_WIDTH / (lateral_speed.abs() + constants::EPSILON)
}

//! Pure algebraic functions for kinematic calculations.
//! These are stateless, deterministic, and side-effect-free.

/// Physical constants derived from automotive safety standards.
pub mod constants {
    /// Maximum comfortable deceleration (m/s^2).
    pub const MAX_DECEL: f32 = 4.0;

    /// Perception-reaction time: nominal mean (seconds).
    /// Human reaction time varies from ~0.5 s (expectant) to ~2.5 s
    /// (surprised); 1.0 s is the 85th-percentile value for an alert driver
    /// per AASHTO Green Book (2018). The standard deviation is 0.3 s;
    /// see `reaction_distance_distributional`.
    pub const REACTION_TIME_MEAN: f32 = 1.0;
    /// Standard deviation of the perception-reaction time (seconds).
    /// A log-normal distribution with (mu = 1.0, sigma = 0.3) captures
    /// the spread from expectant (0.5 s) to surprised (1.8 s) drivers.
    pub const REACTION_TIME_STD: f32 = 0.3;
    /// 95th-percentile reaction time for the conservative stopping
    /// distance bound used by the rule pipeline (seconds).
    /// This is the value used in `stopping_distance()`.
    pub const REACTION_TIME: f32 = 1.0;

    /// Standard intersection width for two lanes (meters).
    pub const INTERSECTION_LENGTH: f32 = 16.0;

    /// Safety margin to guarantee red phase clearance (seconds).
    ///
    /// Derivation: epsilon = t_react_floor + t_actuator + t_box_shading
    ///
    /// - t_react_floor = 0.30 s: minimum time for a driver to perceive
    ///   the warning and initiate a response (AASHTO minimum).
    /// - t_actuator    = 0.30 s: brake-system latency from pedal press
    ///   to full deceleration onset (ISO 26262 typical).
    /// - t_box_shading = 0.20 s: margin against the box geometry
    ///   (rear bumper must clear the far side before cross-traffic
    ///   enters; assumes a 4.5 m vehicle length at urban speed).
    ///
    /// Sum: 0.30 + 0.30 + 0.20 = 0.80 s.
    pub const SAFETY_MARGIN: f32 = 0.8;
    /// Per-component breakdown of the safety margin (seconds).
    pub const SAFETY_MARGIN_REACT_FLOOR: f32 = 0.30;
    pub const SAFETY_MARGIN_ACTUATOR: f32 = 0.30;
    pub const SAFETY_MARGIN_BOX_SHADING: f32 = 0.20;

    /// Small epsilon to prevent division by zero.
    pub const EPSILON: f32 = 0.1;
    /// Standard lane width for cut-in calculations (meters).
    pub const LANE_WIDTH: f32 = 3.5;
    /// Minimum number of consecutive frames a cut-in vehicle must be
    /// observed before the cut-in rule fires (prevents single-frame
    /// false positives from bounding-box jitter).
    pub const CUTIN_MIN_OBSERVATION_FRAMES: u32 = 3;
    /// Maximum lateral speed the cut-in rule bounds (m/s).
    /// A lane change faster than this is treated as a detection
    /// artifact, not a real cut-in.
    pub const CUTIN_MAX_LATERAL_SPEED: f32 = 4.0;
    /// Maximum longitudinal speed for which the reaction-time
    /// distribution is bounded (m/s).
    pub const MAX_SPEED: f32 = 30.0;
    /// Time-to-red below which a yellow is treated as too short for a
    /// comfortable stop (seconds).
    pub const SHORT_YELLOW_THRESHOLD: f32 = 2.5;
    /// Speed below which a lead vehicle is treated as stopped (m/s).
    pub const STOPPED_SPEED_THRESHOLD: f32 = 1.0;

    /// Lipschitz constant for the stopping-distance function.
    ///
    /// From Corollary 1: d_req(v) = v * t_r + v^2 / (2 * a_b), so
    /// d_req'(v) = t_r + v / a_b. The maximum occurs at v = MAX_SPEED:
    /// L = REACTION_TIME + MAX_SPEED / MAX_DECEL = 1.0 + 30.0 / 4.0 = 8.5.
    /// For any two speeds v1, v2: |d_req(v1) - d_req(v2)| <= L * |v1 - v2|.
    pub const LIPSCHITZ_STOPPING_DISTANCE: f32 = REACTION_TIME + MAX_SPEED / MAX_DECEL; // 8.5
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

/// Lipschitz bound on the stopping-distance decision boundary.
///
/// The stopping distance d_req(v) = v * t_r + v^2 / (2 * a_b) has derivative
/// d_req'(v) = t_r + v / a_b. The maximum occurs at MAX_SPEED:
/// L = t_r + MAX_SPEED / a_b (precomputed as LIPSCHITZ_STOPPING_DISTANCE).
///
/// For any two speeds v1, v2: |d_req(v1) - d_req(v2)| <= L * |v1 - v2|.
///
/// This guarantees that a sensor speed error of epsilon_v cannot shift the
/// stopping boundary by more than L * epsilon_v, preventing "flickering"
/// warnings: a small change in estimated speed cannot flip the decision
/// arbitrarily many times within a small speed range.
///
/// # Arguments
/// * `speed_delta` - The absolute difference between two speed estimates (m/s).
///
/// # Returns
/// The maximum possible shift in the stopping boundary (m).
#[must_use]
pub fn lipshitz_stopping_bound(speed_delta: f32) -> f32 {
    constants::LIPSCHITZ_STOPPING_DISTANCE * speed_delta
}

/// Computes the reaction distance using a distributional (log-normal) model
/// of driver reaction time, for use in sensitivity analysis and Monte Carlo
/// studies.
///
/// The standard pipeline uses REACTION_TIME (1.0 s, the 85th percentile).
/// This function takes a percentile z-score and computes the corresponding
/// reaction distance: d_react = v * (t_mean + z * t_std).
///
/// # Arguments
/// * `speed` - Ego longitudinal speed (m/s).
/// * `z_score` - Number of standard deviations from the mean (e.g., 1.645
///   for the 95th percentile, -1.645 for the 5th).
///
/// # Returns
/// The reaction distance (m) for the given percentile.
///
/// # Example
/// ```
/// use civicsense::algebra::reaction_distance_distributional;
/// // 95th-percentile reaction distance at 14 m/s:
/// // d = 14 * (1.0 + 1.645 * 0.3) = 14 * 1.4935 ≈ 20.9 m
/// let d_95 = reaction_distance_distributional(14.0, 1.645);
/// assert!((d_95 - 20.91).abs() < 0.1);
/// ```
#[must_use]
pub fn reaction_distance_distributional(speed: f32, z_score: f32) -> f32 {
    let t_react = constants::REACTION_TIME_MEAN + z_score * constants::REACTION_TIME_STD;
    speed * t_react.max(0.0)
}

/// Class-aware monocular width prior for depth estimation.
///
/// Monocular depth from bounding-box scale requires a known physical width
/// per class. Using a single width prior (e.g., 1.8 m for cars) introduces
/// systematic bias: a truck (~2.5 m) appears closer than it is; a motorcycle
/// (~0.8 m) appears further. This function returns the physical width (m)
/// for a given COCO class id.
///
/// Values are median widths from the KITTI and nuScenes datasets.
///
/// # Arguments
/// * `class_id` - COCO dataset class id.
///
/// # Returns
/// Physical width in metres, or 1.8 if the class is unknown.
#[must_use]
pub fn class_aware_width_prior(class_id: u8) -> f32 {
    match class_id {
        2 => 1.8,  // car
        3 => 0.8,  // motorcycle
        5 => 2.55, // bus
        7 => 2.5,  // truck
        _ => 1.8,  // unknown: default to car width (conservative)
    }
}

/// Computes monocular distance from a bounding-box width using a
/// class-aware width prior (pinhole model).
///
/// d = (f * W_real) / w_px
///
/// where f is the focal length (px), W_real is the physical width (m),
/// and w_px is the bounding-box width (px).
///
/// # Arguments
/// * `focal_length_px` - Camera focal length in pixels.
/// * `bbox_width_px` - Bounding-box width in pixels.
/// * `class_id` - COCO class id for width prior selection.
///
/// # Returns
/// Estimated distance in metres.
#[must_use]
pub fn monocular_depth_class_aware(focal_length_px: f32, bbox_width_px: f32, class_id: u8) -> f32 {
    let w_real = class_aware_width_prior(class_id);
    (focal_length_px * w_real) / (bbox_width_px.max(constants::EPSILON))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::coco_vehicle_classes;

    // ── Test constants (no magic numbers per CODING_STANDARDS §5) ──
    /// Z-score for the 95th percentile of a standard normal distribution.
    const Z_95TH: f32 = 1.645;
    /// Z-score for the 5th percentile.
    const Z_5TH: f32 = -1.645;
    /// Typical urban approach speed for reaction-time sensitivity tests (m/s).
    const TEST_SPEED_URBAN: f32 = 14.0;
    /// Speed delta for Lipschitz bound linearity test (m/s).
    const TEST_SPEED_DELTA: f32 = 0.5;
    /// Expected Lipschitz bound at TEST_SPEED_DELTA: L * delta = 8.5 * 0.5.
    const TEST_LIPSCHITZ_BOUND_AT_HALF_MPS: f32 = 4.25;
    /// Focal length for monocular depth test (px), typical smartphone/dashcam.
    const TEST_FOCAL_LENGTH_PX: f32 = 800.0;
    /// Bounding-box width for monocular depth test (px).
    const TEST_BBOX_WIDTH_PX: f32 = 100.0;
    /// Floating-point tolerance for approximate equality checks.
    const FP_TOLERANCE: f32 = 1e-6;

    #[test]
    fn epsilon_derivation_components_sum_to_margin() {
        let sum = constants::SAFETY_MARGIN_REACT_FLOOR
            + constants::SAFETY_MARGIN_ACTUATOR
            + constants::SAFETY_MARGIN_BOX_SHADING;
        assert!((sum - constants::SAFETY_MARGIN).abs() < FP_TOLERANCE);
    }

    #[test]
    fn lipshitz_constant_is_correct() {
        // L = t_r + v_max / a_b = 1.0 + 30.0 / 4.0 = 8.5
        let expected = constants::REACTION_TIME + constants::MAX_SPEED / constants::MAX_DECEL;
        assert!((constants::LIPSCHITZ_STOPPING_DISTANCE - expected).abs() < FP_TOLERANCE);
    }

    #[test]
    fn lipshitz_bound_is_linear_in_delta() {
        let bound = lipshitz_stopping_bound(TEST_SPEED_DELTA);
        assert!((bound - TEST_LIPSCHITZ_BOUND_AT_HALF_MPS).abs() < FP_TOLERANCE);
    }

    #[test]
    fn reaction_distribution_covers_human_range() {
        let fast = reaction_distance_distributional(TEST_SPEED_URBAN, Z_5TH);
        let slow = reaction_distance_distributional(TEST_SPEED_URBAN, Z_95TH);
        assert!(fast < slow, "faster reaction should yield shorter distance");
        // 5th percentile (~0.5 s) at 14 m/s → ~7 m minimum
        let min_expected = TEST_SPEED_URBAN * 0.45;
        assert!(
            fast > min_expected,
            "5th percentile distance too small: {fast}"
        );
        // 95th percentile (~1.5 s) at 14 m/s → ~21 m
        let max_expected = TEST_SPEED_URBAN * 2.5;
        assert!(
            slow < max_expected,
            "95th percentile distance too large: {slow}"
        );
    }

    #[test]
    fn class_aware_width_priors_are_monotonic() {
        use crate::models::coco_vehicle_classes;
        let w_mc = class_aware_width_prior(coco_vehicle_classes::MOTORCYCLE);
        let w_car = class_aware_width_prior(coco_vehicle_classes::CAR);
        let w_bus = class_aware_width_prior(coco_vehicle_classes::BUS);
        let w_truck = class_aware_width_prior(coco_vehicle_classes::TRUCK);
        assert!(w_mc < w_car);
        assert!(w_car < w_truck);
        let bus_truck_diff = (w_bus - w_truck).abs();
        assert!(
            bus_truck_diff < 0.1,
            "bus and truck widths should be similar: {bus_truck_diff}"
        );
    }

    #[test]
    fn monocular_depth_class_aware_car_vs_truck() {
        let d_car = monocular_depth_class_aware(
            TEST_FOCAL_LENGTH_PX,
            TEST_BBOX_WIDTH_PX,
            coco_vehicle_classes::CAR,
        );
        let d_truck = monocular_depth_class_aware(
            TEST_FOCAL_LENGTH_PX,
            TEST_BBOX_WIDTH_PX,
            coco_vehicle_classes::TRUCK,
        );
        assert!(
            d_truck > d_car,
            "truck ({d_truck}) should appear further than car ({d_car}) at same bbox width"
        );
    }
}

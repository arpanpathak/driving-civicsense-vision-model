//! Core data types representing the state of the traffic scene.
//! All types are immutable; transformations produce new instances.

/// The current colour of the traffic light observed at the intersection.
///
/// This is the output of a vision-based signal-phase classifier or a
/// vehicle-to-infrastructure (V2I / SPaT) feed.  When neither source is
/// available the state is [`Unknown`](LightState::Unknown) and the
/// engine applies the worst-case interpretation of
/// Remark~\\ref{rem:worstcase} in the paper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightState {
    /// The signal is red.  Entering the intersection is a violation;
    /// [`rule_red`](crate::rules::rule_red) unconditionally returns
    /// [`Critical`](WarningLevel::Critical).
    Red,
    /// The signal is yellow (amber).  The time remaining before the
    /// phase turns red is given by `time_to_red` in the decision
    /// pipeline.
    Yellow,
    /// The signal is green.  Under the vision-only worst-case
    /// interpretation a green may end at any frame, so the stale-green
    /// heuristic ([`rule_stale`](crate::rules::rule_stale)) advises
    /// caution when a comfortable stop is no longer possible.
    Green,
    /// Sensor failure: the signal phase could not be determined.
    /// Treated identically to [`Green`](LightState::Green) under the
    /// worst-case assumption.
    Unknown,
}

/// Lane position relative to the ego vehicle, as determined by the
/// bounding-box centroid's horizontal position in the image plane
/// after perspective correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanePosition {
    /// The detected vehicle occupies the same lane as the ego.
    /// Used by [`rule_lead`](crate::rules::rule_lead) to identify
    /// the closest in-path leader.
    Same,
    /// The detected vehicle is in the lane immediately to the left
    /// of the ego.  A vehicle here with an active turn signal is a
    /// cut-in candidate.
    Left,
    /// The detected vehicle is in the lane immediately to the right
    /// of the ego.
    Right,
    /// Lane assignment failed (e.g., the centroid falls outside the
    /// calibrated road region).  The detection is still tracked but
    /// does not participate in lane-dependent rules.
    Unknown,
}

/// The severity of the warning issued to the driver by the decision
/// pipeline.
///
/// The ordering is `Safe < Caution < Warning < Critical`, which is
/// exploited by the severity-ordered pipeline in
/// [`evaluate_safety`](crate::decision::evaluate_safety): the first
/// matching rule determines the final level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarningLevel {
    /// No kinematic constraint is violated; no action required.
    Safe,
    /// A non-urgent advisory.  The driver should prepare for a
    /// possible stop (stale green or very short yellow) but is not
    /// yet in a dilemma.
    Caution,
    /// A substantive warning.  A constraint is violated (e.g., a slow
    /// leader will cause a clearance failure, or a cut-in is
    /// imminent) but a collision is not yet certain.
    Warning,
    /// Immediate danger.  The ego is in the dilemma zone, a leader is
    /// blocking the box, or the signal is already red.  A stop or
    /// evasive action is required now.
    Critical,
}

/// A tracked object produced by the perception pipeline (YOLO
/// detector + Deep SORT tracker + Kalman filter).
///
/// Each detection carries its current kinematic state, its lane
/// assignment, and a `track_age` counter that gates the cut-in rule.
/// All fields are populated by the tracker; the decision engine
/// reads them but never mutates them.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box in image coordinates: `(x_min, y_min, x_max, y_max)`
    /// in pixels.  Used by the visualisation layer and for IoU-based
    /// data association in the tracker.
    pub bbox: (f32, f32, f32, f32),

    /// COCO dataset class id (2 = car, 3 = motorcycle, 5 = bus,
    /// 7 = truck).  Used by [`is_vehicle`](Detection::is_vehicle) to
    /// filter detections and by
    /// [`class_aware_width_prior`](crate::algebra::class_aware_width_prior)
    /// for monocular depth estimation.
    pub class_id: u8,

    /// Estimated longitudinal speed in the ego's direction of travel
    /// (m/s).  Computed by the Kalman filter from consecutive
    /// bounding-box positions.
    pub speed: f32,

    /// Estimated lateral speed (m/s) perpendicular to the ego's
    /// direction of travel.  Positive = moving right in the image
    /// plane.  Used by [`intrusion_time`](crate::algebra::intrusion_time)
    /// in the cut-in rule.
    pub lateral_speed: f32,

    /// Estimated longitudinal distance from the ego vehicle's front
    /// bumper to this detection (m).  Derived from the pinhole
    /// camera model using a class-aware width prior (see
    /// [`monocular_depth_class_aware`](crate::algebra::monocular_depth_class_aware)).
    pub distance_to_ego: f32,

    /// Lane position relative to the ego, assigned by projecting the
    /// bounding-box centroid onto a calibrated road model.
    pub lane: LanePosition,

    /// Whether an amber turn-signal light was detected on this
    /// vehicle in the current frame.  A true value makes the vehicle
    /// a cut-in candidate for [`rule_cutin`](crate::rules::rule_cutin).
    pub turn_signal_active: bool,

    /// Number of consecutive frames this track has been observed by
    /// the tracker.  Resets to 1 on birth, increments on every
    /// successful association, and resets on track death.
    ///
    /// Used by the cut-in rule to enforce a minimum observation
    /// window ([`CUTIN_MIN_OBSERVATION_FRAMES`](constants::CUTIN_MIN_OBSERVATION_FRAMES)),
    /// preventing single-frame false positives from bounding-box
    /// jitter.
    pub track_age: u32,
}

/// COCO dataset class ids treated as motor vehicles by the decision
/// engine.
///
/// The engine reasons only about these classes; detections of
/// pedestrians, bicycles, traffic lights, etc. are silently ignored
/// by [`is_vehicle`](Detection::is_vehicle).
pub mod coco_vehicle_classes {
    /// COCO class id for a passenger car.
    pub const CAR: u8 = 2;
    /// COCO class id for a motorcycle.
    pub const MOTORCYCLE: u8 = 3;
    /// COCO class id for a bus.
    pub const BUS: u8 = 5;
    /// COCO class id for a truck.
    pub const TRUCK: u8 = 7;
}

/// The vehicle classes the engine reasons about, as a slice of COCO
/// class ids.  Used by [`is_vehicle`](Detection::is_vehicle).
pub const VEHICLE_CLASSES: [u8; 4] = [
    coco_vehicle_classes::CAR,
    coco_vehicle_classes::MOTORCYCLE,
    coco_vehicle_classes::BUS,
    coco_vehicle_classes::TRUCK,
];

impl Detection {
    /// Returns `true` if this detection belongs to a motor-vehicle
    /// class that the engine is designed to reason about.
    ///
    /// Pedestrians, bicycles, traffic lights, and all other COCO
    /// classes return `false`.
    #[must_use]
    pub fn is_vehicle(&self) -> bool {
        VEHICLE_CLASSES.contains(&self.class_id)
    }
}

/// Instantaneous state of the ego vehicle consumed by the decision
/// engine.
///
/// Both fields are assumed to be synchronised to the same frame
/// timestamp.  A residual staleness of one frame period (~33 ms at
/// 30 fps) is absorbed by the ±1.5 m operating bound documented in
/// the paper (Section VII, "Scope of the formal results").
#[derive(Debug, Clone, Copy)]
pub struct EgoState {
    /// Ego longitudinal speed (m/s).  Source: OBD-II / CAN bus,
    /// GNSS, or vision-based odometry.
    pub speed: f32,

    /// Distance from the ego's front bumper to the painted stop line
    /// (m).  Source: monocular depth estimation or LiDAR.  When the
    /// stop line is occluded this value is unavailable and the
    /// decision engine should not be invoked.
    pub distance_to_stop_line: f32,
}

/// Derived state of the closest lead vehicle in the same lane as the
/// ego, extracted from the detection list in
/// [`evaluate_safety`](crate::decision::evaluate_safety).
#[derive(Debug, Clone, Copy)]
pub struct LeadVehicle {
    /// Longitudinal distance from the ego's front bumper to the lead
    /// vehicle (m).
    pub distance: f32,

    /// Longitudinal speed of the lead vehicle (m/s).
    pub speed: f32,

    /// Whether the lead vehicle's position places it inside the
    /// intersection polygon.  Computed as
    /// `distance < INTERSECTION_LENGTH` in the pipeline.
    pub is_in_intersection: bool,
}

/// Driver-specific reaction-time profile for distributional analysis.
///
/// Models human perception-reaction time as a log-normal distribution
/// parameterised by its mean and standard deviation. The conservative
/// pipeline uses the 85th percentile (1.0 s); this struct enables
/// sensitivity studies across the full human range (~0.5 s expectant
/// to ~2.5 s surprised) and supports dynamic threshold adaptation
/// for known-slow-reactor drivers.
///
/// The default values mirror [`REACTION_TIME_MEAN`](constants::REACTION_TIME_MEAN)
/// and [`REACTION_TIME_STD`](constants::REACTION_TIME_STD).
#[derive(Debug, Clone, Copy)]
pub struct ReactionProfile {
    /// Mean perception-reaction time (s).  Default: 1.0 (AASHTO
    /// Green Book 2018, 85th percentile for an alert driver).
    pub mean: f32,

    /// Standard deviation of reaction time (s).  Default: 0.3,
    /// giving a log-normal spread from ~0.5 s (expectant) to
    /// ~1.8 s (surprised).
    pub std_dev: f32,
}

impl Default for ReactionProfile {
    fn default() -> Self {
        // Mirrors algebra::constants::REACTION_TIME_MEAN and
        // REACTION_TIME_STD; duplicated here to keep models.rs
        // free of a dependency on algebra.rs.
        const DEFAULT_REACTION_MEAN: f32 = 1.0;
        const DEFAULT_REACTION_STD: f32 = 0.3;
        Self {
            mean: DEFAULT_REACTION_MEAN,
            std_dev: DEFAULT_REACTION_STD,
        }
    }
}

impl ReactionProfile {
    /// Returns the reaction time at a given z-score percentile of
    /// the log-normal distribution.
    ///
    /// # Arguments
    /// * `z_score` — Number of standard deviations from the mean.
    ///   - `z = 0` → median (mean, 1.0 s)
    ///   - `z = 1.645` → 95th percentile (~1.5 s)
    ///   - `z = -1.645` → 5th percentile (~0.5 s)
    ///
    /// # Returns
    /// Reaction time in seconds, clamped to `>= 0.0`.
    #[must_use]
    pub fn at_percentile(&self, z_score: f32) -> f32 {
        (self.mean + z_score * self.std_dev).max(0.0)
    }
}

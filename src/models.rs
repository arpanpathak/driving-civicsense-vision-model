//! Core data types representing the state of the traffic scene.
//! All types are immutable; transformations produce new instances.

/// The current colour of the traffic light.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightState {
    Red,
    Yellow,
    Green,
    Unknown, // Sensor failure.
}

/// Lane position relative to the ego vehicle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanePosition {
    Same,
    Left,
    Right,
    Unknown,
}

/// The severity of the warning issued to the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarningLevel {
    Safe,
    Caution,
    Warning,
    Critical,
}

/// A tracked object from the perception pipeline.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: (f32, f32, f32, f32),
    pub class_id: u8,
    pub speed: f32,
    pub lateral_speed: f32,
    pub distance_to_ego: f32,
    pub lane: LanePosition,
    pub turn_signal_active: bool,
    /// Number of consecutive frames this track has been observed.
    /// Used by the cut-in rule to enforce a minimum observation window
    /// (CUTIN_MIN_OBSERVATION_FRAMES), preventing single-frame false
    /// positives from bounding-box jitter.
    pub track_age: u32,
}

/// COCO dataset class ids treated as motor vehicles by the engine.
/// The engine reasons only about these classes; everything else is ignored.
pub mod coco_vehicle_classes {
    pub const CAR: u8 = 2;
    pub const MOTORCYCLE: u8 = 3;
    pub const BUS: u8 = 5;
    pub const TRUCK: u8 = 7;
}

/// The vehicle classes the engine reasons about (COCO ids).
pub const VEHICLE_CLASSES: [u8; 4] = [
    coco_vehicle_classes::CAR,
    coco_vehicle_classes::MOTORCYCLE,
    coco_vehicle_classes::BUS,
    coco_vehicle_classes::TRUCK,
];

impl Detection {
    /// Returns `true` if the detection belongs to a vehicle class.
    #[must_use]
    pub fn is_vehicle(&self) -> bool {
        VEHICLE_CLASSES.contains(&self.class_id)
    }
}

/// State of the ego vehicle.
#[derive(Debug, Clone, Copy)]
pub struct EgoState {
    pub speed: f32,
    pub distance_to_stop_line: f32,
}

/// Derived state of the closest lead vehicle.
#[derive(Debug, Clone, Copy)]
pub struct LeadVehicle {
    pub distance: f32,
    pub speed: f32,
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
#[derive(Debug, Clone, Copy)]
pub struct ReactionProfile {
    /// Mean perception-reaction time (s).
    pub mean: f32,
    /// Standard deviation of reaction time (s).
    pub std_dev: f32,
}

impl Default for ReactionProfile {
    fn default() -> Self {
        // These values mirror algebra::constants::REACTION_TIME_MEAN
        // and REACTION_TIME_STD; duplicated here to keep models.rs
        // free of a dependency on algebra.rs (CODING_STANDARDS §5).
        const DEFAULT_REACTION_MEAN: f32 = 1.0;
        const DEFAULT_REACTION_STD: f32 = 0.3;
        Self {
            mean: DEFAULT_REACTION_MEAN,
            std_dev: DEFAULT_REACTION_STD,
        }
    }
}

impl ReactionProfile {
    /// Returns the reaction time at a given z-score percentile.
    ///
    /// z = 0     -> median (mean)
    /// z = 1.645 -> 95th percentile (~1.5 s)
    /// z = -1.645 -> 5th percentile (~0.5 s)
    #[must_use]
    pub fn at_percentile(&self, z_score: f32) -> f32 {
        (self.mean + z_score * self.std_dev).max(0.0)
    }
}

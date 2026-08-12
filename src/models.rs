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
}

impl Detection {
    /// Returns `true` if the detection belongs to a vehicle class.
    #[must_use]
    pub fn is_vehicle(&self) -> bool {
        matches!(self.class_id, 2 | 3 | 5 | 7)
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

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

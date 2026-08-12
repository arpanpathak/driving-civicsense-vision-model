//! Core data types representing the state of the traffic scene.
//!
//! Every type in this module is **immutable after construction**;
//! transformations produce new instances.  The decision engine reads
//! these types but never mutates them — this is a deliberate design
//! choice that eliminates entire classes of concurrency bugs and
//! makes the pipeline trivially auditable for ISO 26262 traceability.
//!
//! # Type hierarchy
//!
//! ```text
//! Inputs (from perception)          Decision layer             Output
//! ─────────────────────           ────────────────            ──────
//! Detection ──────────────┐
//! EgoState ───────────────┼──→ evaluate_safety() ──→ WarningLevel
//! LightState ─────────────┤        │
//! time_to_red: f32 ───────┘        ├── rule_red()
//!                                  ├── rule_dilemma()
//!                                  ├── rule_lead()
//!                                  ├── rule_cutin()
//!                                  ├── rule_yellow()
//!                                  └── rule_stale()
//! ```
//!
//! # Paper cross-reference
//!
//! Every type in this module corresponds to notation in the paper
//! (Table I, Notation and physical constants).  See
//! [`docs/index.html`](https://arpanpathak.github.io/driving-civicsense-vision-model/)
//! for the rendered paper and
//! [`research_paper/paper.pdf`](https://raw.githubusercontent.com/arpanpathak/driving-civicsense-vision-model/main/research_paper/paper.pdf)
//! for the PDF.

// ── LightState ──────────────────────────────────────────────────────

/// The current colour (phase) of the traffic light at the intersection
/// being approached.
///
/// # Source
///
/// This value is the output of either:
///
/// - A **vision-based signal-phase classifier** (e.g., a YOLO subclass
///   trained on red/yellow/green traffic lights, or a colour-threshold
///   heuristic on the region above the stop line), or
/// - A **vehicle-to-infrastructure (V2I / SPaT) feed** that broadcasts
///   the current phase and time-to-red over DSRC or C-V2X.
///
/// When **neither source is available** the state is
/// [`Unknown`](LightState::Unknown), and the decision engine applies
/// the **vision-only worst-case interpretation**: every green is treated
/// as potentially ending at the current frame (Remark in Section IV-G
/// of the paper).
///
/// # Correctness invariants
///
/// - [`Red`](LightState::Red) must **always** produce
///   [`Critical`](WarningLevel::Critical) — this is enforced by
///   [`rule_red`](crate::rules::rule_red) and verified by the
///   `red_light_is_always_critical` property test.
/// - [`Yellow`](LightState::Yellow) and [`Green`](LightState::Green)
///   require a valid `time_to_red` to make a decision; without it, the
///   worst-case geometry is assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightState {
    /// **Red light.**  Entering the intersection is a traffic violation
    /// regardless of speed or distance.  [`rule_red`](crate::rules::rule_red)
    /// unconditionally returns [`Critical`](WarningLevel::Critical).
    /// This is the highest-priority rule in the pipeline; it cannot be
    /// masked by any other criterion.
    Red,

    /// **Yellow (amber) light.**  The time remaining before the phase
    /// turns red is given by `time_to_red` in the decision pipeline
    /// (see [`evaluate_safety`](crate::decision::evaluate_safety)).
    ///
    /// Under a yellow signal, three rules may fire:
    /// 1. [`rule_dilemma`](crate::rules::rule_dilemma) if the ego is
    ///    in the dilemma zone (cannot stop, cannot clear).
    /// 2. [`rule_lead`](crate::rules::rule_lead) if a leader blocks
    ///    the intersection.
    /// 3. [`rule_yellow`](crate::rules::rule_yellow) if the remaining
    ///    time is below [`SHORT_YELLOW_THRESHOLD`](crate::algebra::constants::SHORT_YELLOW_THRESHOLD).
    Yellow,

    /// **Green light.**  The driver expects to proceed, but a stale
    /// green can transition to yellow at any frame.  Under the
    /// vision-only worst-case interpretation (no V2I / SPaT feed),
    /// [`rule_stale`](crate::rules::rule_stale) fires when the ego is
    /// beyond the comfortable-stop envelope, advising Caution.
    ///
    /// Three rules may still fire on green:
    /// 1. [`rule_lead`](crate::rules::rule_lead) if a leader is stopped
    ///    in the box or cannot clear.
    /// 2. [`rule_cutin`](crate::rules::rule_cutin) if an adjacent vehicle
    ///    signals and intrudes.
    /// 3. [`rule_stale`](crate::rules::rule_stale) for the worst-case
    ///    advisory.
    Green,

    /// **Sensor failure.**  The signal phase could not be determined
    /// (e.g., the stop-light region is occluded, the classifier returned
    /// below-threshold confidence, or the V2I feed timed out).
    ///
    /// Treated **identically to [`Green`](LightState::Green)** under the
    /// worst-case assumption: every frame is treated as a potential
    /// phase transition.  This is the safe/conservative choice — it
    /// may increase false-positive Caution advisories but will never
    /// miss a real yellow.
    Unknown,
}

// ── LanePosition ─────────────────────────────────────────────────────

/// Lane position of a detected vehicle relative to the ego, assigned
/// by projecting the bounding-box centroid onto a calibrated road
/// model after perspective correction.
///
/// # Assignment method
///
/// The lane is determined by the horizontal position of the bounding-box
/// centroid in the image plane, mapped to world coordinates via the
/// pinhole camera model, then bucketed into left / same / right relative
/// to the ego's lane centerline.
///
/// # Correctness invariants
///
/// - Only [`Same`](LanePosition::Same) vehicles participate in
///   [`rule_lead`](crate::rules::rule_lead).
/// - Only [`Left`](LanePosition::Left) or [`Right`](LanePosition::Right)
///   vehicles with an active turn signal participate in
///   [`rule_cutin`](crate::rules::rule_cutin).
/// - [`Unknown`](LanePosition::Unknown) vehicles are tracked but never
///   trigger lane-dependent rules — a deliberate choice to avoid false
///   positives from misclassified lane positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanePosition {
    /// **Same lane.**  The detected vehicle occupies the same
    /// longitudinal lane as the ego.  Used by
    /// [`rule_lead`](crate::rules::rule_lead) to identify the closest
    /// in-path leader for the following-blockage check (Theorem 5).
    Same,

    /// **Left lane.**  The detected vehicle is in the lane immediately
    /// to the left of the ego.  If it has an active turn signal and
    /// passes the latency filter (≥
    /// [`CUTIN_MIN_OBSERVATION_FRAMES`](crate::algebra::constants::CUTIN_MIN_OBSERVATION_FRAMES)
    /// frames), it becomes a cut-in candidate for
    /// [`rule_cutin`](crate::rules::rule_cutin).
    Left,

    /// **Right lane.**  The detected vehicle is in the lane immediately
    /// to the right of the ego.  Treated symmetrically to
    /// [`Left`](LanePosition::Left) by the cut-in rule.
    Right,

    /// **Lane assignment failed.**  The centroid falls outside the
    /// calibrated road region (e.g., vehicle on the shoulder, or
    /// camera pitch error).  The detection is still tracked and its
    /// kinematic estimates are maintained, but it does **not**
    /// participate in `rule_lead` or `rule_cutin`.  This is the safe
    /// choice: a misclassified lane is worse than a missed rule fire.
    Unknown,
}

// ── WarningLevel ─────────────────────────────────────────────────────

/// The severity of the warning issued to the driver by the decision
/// pipeline.
///
/// # Ordering
///
/// `Safe < Caution < Warning < Critical`
///
/// This ordering is derived from [`PartialOrd`] and exploited by
/// [`evaluate_safety`](crate::decision::evaluate_safety): the pipeline
/// evaluates rules in descending severity order, and `find_map` returns
/// the first match — which is guaranteed to be the most severe.
///
/// # Design philosophy
///
/// The pipeline is **recall-first**: it biases toward higher severity
/// to minimise false negatives (missed blockages).  A false positive
/// (unnecessary warning) erodes trust; a false negative can be fatal.
/// The severity-ordered composition ensures that a Critical rule is
/// never masked by a lower-severity rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarningLevel {
    /// **Safe.**  No kinematic constraint is violated; no action
    /// required.  Either the ego is comfortably within the stopping
    /// envelope (d_s ≤ d_req) and the light is green, or the ego can
    /// safely clear the intersection before the red phase.
    Safe,

    /// **Caution.**  A non-urgent advisory.  The driver should prepare
    /// for a possible stop but is not yet in immediate danger.  Fired
    /// by two rules:
    /// - [`rule_yellow`](crate::rules::rule_yellow): yellow with very
    ///   little time remaining (< 2.5 s).
    /// - [`rule_stale`](crate::rules::rule_stale): stale green where
    ///   the ego is beyond the comfortable-stop envelope.
    Caution,

    /// **Warning.**  A substantive alert.  A kinematic constraint is
    /// violated but a collision is not certain.  Fired by two rules:
    /// - [`rule_lead`](crate::rules::rule_lead) sub-rule 3b: following
    ///   a slow leader will cause a clearance failure.
    /// - [`rule_cutin`](crate::rules::rule_cutin): an adjacent vehicle
    ///   with an active turn signal will intrude before the light
    ///   changes, invalidating the stopping distance.
    Warning,

    /// **Critical.**  Immediate danger.  The driver must stop or take
    /// evasive action **now**.  Fired by three rules:
    /// - [`rule_red`](crate::rules::rule_red): light is already red.
    /// - [`rule_dilemma`](crate::rules::rule_dilemma): ego can neither
    ///   stop before the line nor clear the box before the red phase.
    /// - [`rule_lead`](crate::rules::rule_lead) sub-rule 3a: a leader
    ///   is already stopped inside the intersection and the ego cannot
    ///   stop behind it.
    Critical,
}

// ── Detection ────────────────────────────────────────────────────────

/// A tracked object produced by the perception pipeline.
///
/// This is the output of **YOLO detector → NMS → Deep SORT tracker →
/// Kalman filter** — it carries the current kinematic state, lane
/// assignment, turn-signal status, and a **track age** counter that
/// gates the cut-in rule's latency filter.
///
/// # Lifecycle
///
/// 1. **Birth**: a new YOLO detection is associated with a new track
///    ID; `track_age = 1`.
/// 2. **Update**: on every successful frame-to-frame association, the
///    Kalman filter updates speed/lateral_speed/distance and
///    `track_age` increments.
/// 3. **Death**: after `max_age` consecutive missed associations, the
///    track is pruned.
///
/// # Immutability
///
/// All fields are populated by the tracker.  The decision engine reads
/// them but never mutates them.  This is enforced by convention (the
/// struct has no `pub mut` methods); a future revision may enforce it
/// at the type level.
#[derive(Debug, Clone)]
pub struct Detection {
    /// **Bounding box in image coordinates.**
    ///
    /// Tuple `(x_min, y_min, x_max, y_max)` in **pixels**, with origin
    /// at the top-left corner of the frame.
    ///
    /// Used by:
    /// - The **visualisation layer** (`utils::visualization`) to draw
    ///   boxes on the debug overlay.
    /// - The **IoU-based data association** in the Deep SORT tracker
    ///   (`tracking::deep_sort`).
    /// - The **pinhole distance estimator** (`utils::geometry`) to
    ///   convert pixel width to metric distance.
    ///
    /// Typical ranges (1280×720 frame, urban scene):
    /// - Near vehicle (5 m): ~300–500 px wide
    /// - Far vehicle (60 m): ~40–80 px wide
    pub bbox: (f32, f32, f32, f32),

    /// **COCO dataset class id.**
    ///
    /// The engine reasons only about motor vehicles (see
    /// [`coco_vehicle_classes`] and [`VEHICLE_CLASSES`]); detections
    /// of pedestrians, bicycles, traffic lights, etc. are silently
    /// ignored by [`is_vehicle`](Detection::is_vehicle).
    ///
    /// Known values:
    /// | Class ID | Label | Physical width (m) |
    /// |----------|-------|-------------------|
    /// | 2 | Car | 1.80 |
    /// | 3 | Motorcycle | 0.80 |
    /// | 5 | Bus | 2.55 |
    /// | 7 | Truck | 2.50 |
    ///
    /// The physical width is used by
    /// [`class_aware_width_prior`](crate::algebra::class_aware_width_prior)
    /// for monocular depth estimation — using a class-agnostic width
    /// prior introduces systematic bias (a truck appears closer than
    /// it is; a motorcycle appears further).
    pub class_id: u8,

    /// **Estimated longitudinal speed (m/s).**
    ///
    /// Speed in the ego's direction of travel.  Computed by the Kalman
    /// filter from consecutive bounding-box centroid displacements,
    /// converted to metric units via the pinhole model.
    ///
    /// Typical urban range: 0–20 m/s (0–72 km/h, 0–45 mph).
    /// Values outside this range are clamped by the tracker but not
    /// rejected by the decision engine (the kinematic theorems are
    /// valid for all non-negative speeds).
    pub speed: f32,

    /// **Estimated lateral speed (m/s).**
    ///
    /// Speed **perpendicular** to the ego's direction of travel.
    /// Positive = moving right in the image plane; negative = moving
    /// left.
    ///
    /// Used exclusively by [`intrusion_time`](crate::algebra::intrusion_time)
    /// in [`rule_cutin`](crate::rules::rule_cutin) to estimate when the
    /// adjacent vehicle will cross the lane boundary.
    ///
    /// The cut-in rule caps this at
    /// [`CUTIN_MAX_LATERAL_SPEED`](crate::algebra::constants::CUTIN_MAX_LATERAL_SPEED)
    /// (4.0 m/s); values above this are treated as detection artifacts
    /// (a vehicle cannot physically change lanes faster than ~4 m/s
    /// laterally).
    pub lateral_speed: f32,

    /// **Estimated longitudinal distance from ego to detection (m).**
    ///
    /// Distance from the ego vehicle's **front bumper** to the detected
    /// vehicle, along the ego's longitudinal axis.
    ///
    /// Derived from the **pinhole camera model**:
    ///
    /// ```text
    /// distance = (focal_length_px × physical_width_m) / bbox_width_px
    /// ```
    ///
    /// The physical width is chosen by
    /// [`class_aware_width_prior`](crate::algebra::class_aware_width_prior)
    /// based on `class_id`.  See
    /// [`monocular_depth_class_aware`](crate::algebra::monocular_depth_class_aware).
    ///
    /// **Accuracy:** ±1.5 m at urban speeds (one-pixel jitter at 30 fps
    /// plus calibration tolerance, per Section VII of the paper).  This
    /// bound reserves the slack for perception error; the decision engine
    /// itself treats the input as exact.
    pub distance_to_ego: f32,

    /// **Lane position relative to the ego.**
    ///
    /// Assigned by projecting the bounding-box centroid onto a calibrated
    /// road model.  See [`LanePosition`] for the semantics of each variant.
    pub lane: LanePosition,

    /// **Turn-signal (amber blinker) detection status.**
    ///
    /// `true` if an amber turn-signal light was detected on this vehicle
    /// in the current frame.  Detection is performed by a YOLO subclass
    /// or a colour-threshold heuristic on the vehicle's bounding-box
    /// region.
    ///
    /// A `true` value makes the vehicle a **cut-in candidate** for
    /// [`rule_cutin`](crate::rules::rule_cutin), provided it also passes
    /// the latency filter (`track_age ≥ CUTIN_MIN_OBSERVATION_FRAMES`)
    /// and the lateral-speed cap.
    ///
    /// **False positive risk:** a single-frame bounding-box jitter can
    /// simulate a spurious `turn_signal_active` flag on a stationary
    /// parked car.  This is why the cut-in rule enforces a 3-frame
    /// minimum observation window.
    pub turn_signal_active: bool,

    /// **Track age — consecutive frames this track has been observed.**
    ///
    /// Lifecycle:
    /// - **Birth**: `track_age = 1` on first association.
    /// - **Update**: incremented on every successful frame-to-frame
    ///   association.
    /// - **Death**: reset when the track is pruned after `max_age`
    ///   consecutive missed associations.
    ///
    /// Used by [`rule_cutin`](crate::rules::rule_cutin) to enforce a
    /// minimum observation window of
    /// [`CUTIN_MIN_OBSERVATION_FRAMES`](crate::algebra::constants::CUTIN_MIN_OBSERVATION_FRAMES)
    /// (3 frames).  This prevents single-frame false positives from
    /// bounding-box jitter that could simulate a spurious turn signal
    /// on a stationary parked car or a one-frame detection artifact.
    ///
    /// A track with `track_age < CUTIN_MIN_OBSERVATION_FRAMES` is
    /// **never** considered by the cut-in rule, regardless of its
    /// `turn_signal_active` status.
    pub track_age: u32,
}

// ── COCO vehicle classes ─────────────────────────────────────────────

/// COCO dataset class ids treated as **motor vehicles** by the
/// decision engine.
///
/// The engine reasons **only** about these four classes; detections of
/// pedestrians (class 0), bicycles (1), traffic lights (9), stop signs
/// (11), and all other COCO labels are silently ignored by
/// [`is_vehicle`](Detection::is_vehicle).
///
/// # Why these four?
///
/// These are the COCO classes that represent motorised road vehicles
/// capable of blocking an intersection or cutting into the ego's lane.
/// Motorcycles are included because they occupy a lane and can trigger
/// the lead-vehicle rule, unlike bicycles which can be safely passed.
pub mod coco_vehicle_classes {
    /// COCO class id 2: **passenger car** (physical width ~1.80 m).
    pub const CAR: u8 = 2;
    /// COCO class id 3: **motorcycle** (physical width ~0.80 m).
    pub const MOTORCYCLE: u8 = 3;
    /// COCO class id 5: **bus** (physical width ~2.55 m).
    pub const BUS: u8 = 5;
    /// COCO class id 7: **truck** (physical width ~2.50 m).
    pub const TRUCK: u8 = 7;
}

/// The vehicle classes the engine reasons about, as a constant array
/// of COCO class ids.
///
/// Used by [`is_vehicle`](Detection::is_vehicle) via
/// [`contains`](slice::contains).  The order is arbitrary; the array
/// exists solely for the O(4) membership check.
pub const VEHICLE_CLASSES: [u8; 4] = [
    coco_vehicle_classes::CAR,
    coco_vehicle_classes::MOTORCYCLE,
    coco_vehicle_classes::BUS,
    coco_vehicle_classes::TRUCK,
];

impl Detection {
    /// Returns `true` if this detection is a motor vehicle that the
    /// engine is designed to reason about.
    ///
    /// Checks `class_id` against [`VEHICLE_CLASSES`].  Pedestrians
    /// (0), bicycles (1), traffic lights (9), stop signs (11), and
    /// all other COCO classes return `false`.
    ///
    /// This is called by:
    /// - The pipeline's lead-vehicle extraction
    ///   (`detections.iter().find(|d| d.is_vehicle() && d.lane == Same)`)
    /// - The cut-in rule's filter
    ///   (`detections.iter().filter(|d| d.is_vehicle() && ...)`)
    #[must_use]
    pub fn is_vehicle(&self) -> bool {
        VEHICLE_CLASSES.contains(&self.class_id)
    }
}

// ── EgoState ─────────────────────────────────────────────────────────

/// **Instantaneous snapshot of the ego vehicle** consumed by the
/// decision engine.
///
/// # Synchronisation requirement
///
/// Both fields must refer to the **same frame timestamp**.  The engine
/// treats the input vector as a synchronised snapshot; a residual
/// staleness of one frame period (~33 ms at 30 fps) is absorbed by the
/// ±1.5 m operating bound documented in Section VII of the paper
/// ("Scope of the formal results").
///
/// # Invariants
///
/// - `speed >= 0.0` — negative speed is physically impossible for
///   forward-facing perception.
/// - `distance_to_stop_line >= 0.0` — the stop line cannot be behind
///   the ego (crossing it means the ego is already in the intersection,
///   at which point the decision engine should not be invoked).
///
/// These invariants are **not enforced at the type level** (the struct
/// uses bare `f32`).  Moving them into the type system (e.g., a
/// `NonNegativeF32` newtype) is planned future work.
#[derive(Debug, Clone, Copy)]
pub struct EgoState {
    /// **Ego longitudinal speed (m/s).**
    ///
    /// Source options (in order of accuracy):
    /// 1. **OBD-II / CAN bus** — direct wheel-speed sensor readout,
    ///    ±0.1 m/s accuracy, typically 10–100 Hz.
    /// 2. **GNSS (RTK)** — ±0.02 m/s, requires clear sky view.
    /// 3. **Vision-based odometry** — derived from optical flow or
    ///    consecutive-frame feature tracking; accuracy depends on
    ///    calibration quality and frame rate.
    ///
    /// Typical urban approach speeds: 5–20 m/s (18–72 km/h).
    /// The kinematic theorems are valid for all `speed >= 0`.
    ///
    /// This value is consumed by:
    /// - [`stopping_distance`](crate::algebra::stopping_distance) —
    ///   computes d_req = v·t_r + v²/(2·a_b).
    /// - [`clearance_time`](crate::algebra::clearance_time) —
    ///   computes t_c = (d_s + L_i) / v.
    /// - [`rule_lead`](crate::rules::rule_lead) — compares ego
    ///   stopping distance against leader distance.
    pub speed: f32,

    /// **Distance from ego's front bumper to the painted stop line (m).**
    ///
    /// Source options (in order of accuracy):
    /// 1. **LiDAR** — direct range measurement, ±2–3 cm.
    /// 2. **Radar** — ±0.1 m, requires a reflective stop-line target.
    /// 3. **Monocular depth estimation** — derived from the pinhole
    ///    camera model using known stop-line geometry (width × focal
    ///    length / pixel width).  Accuracy ±1.5 m per the paper's
    ///    operating bound.
    ///
    /// **When the stop line is occluded** (e.g., by a large vehicle),
    /// this value is unavailable and the decision engine **should not
    /// be invoked** — this is documented as a known vulnerability
    /// (Table V, row 5: Occlusion) in the paper.
    ///
    /// This value is consumed by:
    /// - [`rule_dilemma`](crate::rules::rule_dilemma) — the core
    ///   stopping-clearance conjunction.
    /// - [`clearance_time`](crate::algebra::clearance_time) —
    ///   used by both the dilemma and lead-vehicle rules.
    pub distance_to_stop_line: f32,
}

// ── LeadVehicle ──────────────────────────────────────────────────────

/// **Derived state of the closest lead vehicle** in the same lane as
/// the ego.
///
/// Extracted from the detection list in
/// [`evaluate_safety`](crate::decision::evaluate_safety) by finding
/// the first [`Detection`] with
/// `is_vehicle() && lane == LanePosition::Same`.
///
/// # Extraction logic
///
/// The pipeline picks the **closest** same-lane vehicle (the one with
/// minimum `distance_to_ego`), not the fastest or the largest.  This
/// is the correct choice for the following-blockage check: if the
/// closest leader cannot clear, the ego is trapped regardless of what
/// vehicles further ahead are doing.
#[derive(Debug, Clone, Copy)]
pub struct LeadVehicle {
    /// **Longitudinal distance from ego's front bumper to the lead
    /// vehicle (m).**
    ///
    /// Copied from [`Detection::distance_to_ego`].  Used in:
    /// - Sub-rule 3a: `lead.distance < d_req && lead.is_in_intersection`
    ///   → Critical (leader stopped in the box, ego cannot stop behind).
    /// - Sub-rule 3b: `clearance_time(lead.distance, lead.speed) >=
    ///   t_red - ε` → Warning (following the leader causes clearance
    ///   failure).
    pub distance: f32,

    /// **Longitudinal speed of the lead vehicle (m/s).**
    ///
    /// Copied from [`Detection::speed`].  A value below
    /// [`STOPPED_SPEED_THRESHOLD`](crate::algebra::constants::STOPPED_SPEED_THRESHOLD)
    /// (1.0 m/s) triggers sub-rule 3a (Critical — leader stopped in
    /// box).  Higher values are used in sub-rule 3b's effective
    /// clearance-time calculation.
    pub speed: f32,

    /// **Whether the lead vehicle is inside the intersection polygon.**
    ///
    /// Computed as `distance < INTERSECTION_LENGTH` (16 m) in the
    /// pipeline.  This is a **geometric approximation** — the true
    /// intersection boundary may be irregular — but 16 m is the
    /// standard width for a two-lane urban intersection and is
    /// sufficient for the kinematic check.
    ///
    /// Required by sub-rule 3a: a leader is only "blocking the box"
    /// if it is both stopped **and** inside the intersection.
    pub is_in_intersection: bool,
}

// ── ReactionProfile ──────────────────────────────────────────────────

/// **Driver-specific reaction-time profile** for distributional
/// (log-normal) analysis of human perception-reaction time.
///
/// # Motivation
///
/// The conservative pipeline uses a **point estimate** of 1.0 s (the
/// 85th percentile per AASHTO Green Book 2018).  But human reaction
/// time is a **distribution**, ranging from ~0.5 s (expectant, alert
/// driver) to ~2.5 s (surprised, distracted driver).  Using a single
/// point value makes the "safe stop" guarantee conditional on the
/// driver being at or above the 85th percentile.
///
/// This struct enables:
/// - **Sensitivity studies**: run the Monte Carlo simulation with
///   reaction times sampled from the distribution to see how the
///   false-negative rate varies with driver alertness.
/// - **Dynamic threshold adaptation**: for a driver whose measured
///   reaction time is known to be slow (e.g., from steering-wheel
///   tap-response measurements), the pipeline can substitute a higher
///   percentile, widening the safety envelope without invalidating
///   the kinematic theorems (future work, Section VII of the paper).
///
/// # Default values
///
/// | Field | Value | Source |
/// |-------|-------|--------|
/// | `mean` | 1.0 s | AASHTO Green Book 2018, 85th percentile |
/// | `std_dev` | 0.3 s | Log-normal fit to published reaction-time histograms |
///
/// These mirror
/// [`REACTION_TIME_MEAN`](crate::algebra::constants::REACTION_TIME_MEAN)
/// and
/// [`REACTION_TIME_STD`](crate::algebra::constants::REACTION_TIME_STD).
#[derive(Debug, Clone, Copy)]
pub struct ReactionProfile {
    /// **Mean perception-reaction time (s).**
    ///
    /// Default: 1.0 s — the 85th percentile for an alert driver per
    /// AASHTO Green Book (2018).  This is the value used by
    /// [`stopping_distance`](crate::algebra::stopping_distance) in the
    /// standard pipeline.
    ///
    /// Range: 0.5–2.5 s covers the human population from expectant to
    /// surprised.
    pub mean: f32,

    /// **Standard deviation of reaction time (s).**
    ///
    /// Default: 0.3 s, giving a log-normal spread:
    /// - 5th percentile (`z = -1.645`): ~0.5 s (very alert)
    /// - Median (`z = 0`): 1.0 s
    /// - 95th percentile (`z = +1.645`): ~1.5 s (slow reactor)
    ///
    /// The 95th-percentile value adds `0.5 × speed` metres to the
    /// stopping envelope compared to the median driver.
    pub std_dev: f32,
}

impl Default for ReactionProfile {
    /// Returns the **default reaction profile** matching the
    /// conservative pipeline's point estimate.
    ///
    /// Mirrors
    /// [`REACTION_TIME_MEAN`](crate::algebra::constants::REACTION_TIME_MEAN)
    /// and
    /// [`REACTION_TIME_STD`](crate::algebra::constants::REACTION_TIME_STD);
    /// duplicated here as named constants to keep `models.rs` free
    /// of a compile-time dependency on `algebra.rs` (per
    /// CODING_STANDARDS §5).
    fn default() -> Self {
        const DEFAULT_REACTION_MEAN: f32 = 1.0;
        const DEFAULT_REACTION_STD: f32 = 0.3;
        Self {
            mean: DEFAULT_REACTION_MEAN,
            std_dev: DEFAULT_REACTION_STD,
        }
    }
}

impl ReactionProfile {
    /// Returns the reaction time at a given **z-score percentile**
    /// of the log-normal distribution.
    ///
    /// # Formula
    ///
    /// ```text
    /// t_react(z) = mean + z × std_dev   (clamped to ≥ 0)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `z_score` — Number of standard deviations from the mean:
    ///   - `z = 0` → median (mean, 1.0 s)
    ///   - `z = +1.645` → 95th percentile (~1.5 s)
    ///   - `z = -1.645` → 5th percentile (~0.5 s)
    ///   - `z = +2.326` → 99th percentile (~1.7 s)
    ///
    /// # Returns
    ///
    /// Reaction time in seconds, clamped to `>= 0.0` (negative
    /// reaction times are physically impossible).
    #[must_use]
    pub fn at_percentile(&self, z_score: f32) -> f32 {
        (self.mean + z_score * self.std_dev).max(0.0)
    }
}

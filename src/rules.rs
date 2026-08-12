//! Decision rules. Each function implements one criterion of
//! Section 4 of the paper. All functions are pure, deterministic,
//! and return `Option<WarningLevel>` to facilitate composition in
//! the severity-ordered pipeline.
//!
//! # Rule numbering
//!
//! | Rule | Function | Condition | Severity |
//! |------|----------|-----------|----------|
//! | 1 | [`rule_red`] | Light is red | Critical |
//! | 2 | [`rule_dilemma`] | Cannot stop ∧ cannot clear | Critical |
//! | 3 | [`rule_lead`] | Leader blocks the box | Critical/Warning |
//! | 4 | [`rule_cutin`] | Adjacent vehicle cuts in | Warning |
//! | 5 | [`rule_yellow`] | Short yellow (< 2.5 s) | Caution |
//! | 6 | [`rule_stale`] | Stale green, cannot stop | Caution |

use crate::algebra::*;
use crate::models::*;

/// Rule 1 (Section 4.6): Red-light rule.
///
/// A red light implies Critical, unconditionally.  This is the
/// highest-severity rule and must appear first in the pipeline so
/// that no other rule can mask a red signal.
///
/// # Arguments
/// * `light` — The current traffic-light phase observed at the
///   intersection.  Only [`LightState::Red`] triggers this rule.
///
/// # Returns
/// `Some(Critical)` if the light is red; `None` otherwise.
#[must_use]
pub fn rule_red(light: LightState) -> Option<WarningLevel> {
    match light {
        LightState::Red => Some(WarningLevel::Critical),
        _ => None,
    }
}

/// Rule 2 (Theorem 4): Dilemma zone rule.
///
/// The core stopping-clearance conjunction.  Returns Critical when
/// the ego can neither stop before the stop line nor clear the
/// intersection before the signal turns red.
///
/// # Arguments
/// * `ego` — Instantaneous state of the ego vehicle (speed and
///   distance to the stop line, assumed synchronised to the same
///   frame).
/// * `time_to_red` — Seconds remaining before the signal turns red
///   (from V2I/SPaT or a vision-based countdown detector).
///
/// # Returns
/// `Some(Critical)` if both the stopping and clearance conditions
/// fail; `None` otherwise.
#[must_use]
pub fn rule_dilemma(ego: &EgoState, time_to_red: f32) -> Option<WarningLevel> {
    let d_req = stopping_distance(ego.speed);
    let t_c = clearance_time(ego.distance_to_stop_line, ego.speed);

    let cannot_stop = ego.distance_to_stop_line <= d_req;
    let cannot_clear = t_c >= (time_to_red - constants::SAFETY_MARGIN);

    match (cannot_stop, cannot_clear) {
        (true, true) => Some(WarningLevel::Critical),
        _ => None,
    }
}

/// Rule 3 (Theorem 5): Lead vehicle rule.
///
/// Checks two sub-conditions:
///
/// *Sub-rule 3a*: the leader is stopped (speed below
/// [`STOPPED_SPEED_THRESHOLD`](constants::STOPPED_SPEED_THRESHOLD)),
/// already inside the intersection, and the ego cannot stop behind
/// it → **Critical**.
///
/// *Sub-rule 3b*: following the leader causes a clearance failure
/// (the effective clearance time exceeds the time-to-red minus the
/// safety margin) and the ego cannot stop behind the leader →
/// **Warning**.
///
/// # Arguments
/// * `ego_speed` — Ego longitudinal speed (m/s), from
///   [`EgoState::speed`].
/// * `lead` — Derived state of the closest lead vehicle in the same
///   lane, extracted from the detection list by the pipeline.
/// * `time_to_red` — Seconds remaining before the signal turns red.
///
/// # Returns
/// `Some(Critical)` if sub-rule 3a fires; `Some(Warning)` if
/// sub-rule 3b fires; `None` if neither applies.
#[must_use]
pub fn rule_lead(ego_speed: f32, lead: &LeadVehicle, time_to_red: f32) -> Option<WarningLevel> {
    let d_req = stopping_distance(ego_speed);

    // Sub-rule 3a: leader is stopped and already inside the intersection.
    if lead.speed < constants::STOPPED_SPEED_THRESHOLD
        && lead.distance < d_req
        && lead.is_in_intersection
    {
        return Some(WarningLevel::Critical);
    }

    // Sub-rule 3b: following the leader causes a clearance failure.
    let t_eff = clearance_time(lead.distance, lead.speed);
    if t_eff >= (time_to_red - constants::SAFETY_MARGIN) && lead.distance < d_req {
        return Some(WarningLevel::Warning);
    }

    None
}

/// Rule 4 (Theorem 6): Cut-in rule.
///
/// Detects adjacent vehicles with active turn signals that will
/// intrude into the ego's lane before the light changes.
///
/// # Latency condition
///
/// A cut-in vehicle must be observed for at least
/// [`CUTIN_MIN_OBSERVATION_FRAMES`](constants::CUTIN_MIN_OBSERVATION_FRAMES)
/// consecutive frames before the rule fires — this prevents
/// single-frame false positives from bounding-box jitter mimicking a
/// `turn_signal_active` flag on a stationary parked car or detection
/// artifact.
///
/// Additionally, the lateral speed is capped at
/// [`CUTIN_MAX_LATERAL_SPEED`](constants::CUTIN_MAX_LATERAL_SPEED);
/// values above this are treated as detection artifacts (a vehicle
/// cannot physically change lanes faster than ~4 m/s laterally).
///
/// # Arguments
/// * `detections` — The current frame's tracked detections, each
///   carrying lane assignment, turn-signal status, lateral speed,
///   and track age.
/// * `ego_speed` — Ego longitudinal speed (m/s), used to compute
///   the required stopping distance against which the intruder's
///   distance is compared.
/// * `time_to_red` — Seconds remaining before the signal turns red.
///
/// # Returns
/// `Some(Warning)` if any adjacent vehicle with an active signal
/// passes both the latency filter and the kinematic intrusion check;
/// `None` otherwise.
#[must_use]
pub fn rule_cutin(
    detections: &[Detection],
    ego_speed: f32,
    time_to_red: f32,
) -> Option<WarningLevel> {
    let d_req = stopping_distance(ego_speed);

    detections
        .iter()
        .filter(|d| {
            d.is_vehicle()
                && d.turn_signal_active
                && d.lane != LanePosition::Same
                && d.track_age >= constants::CUTIN_MIN_OBSERVATION_FRAMES
                && d.lateral_speed.abs() <= constants::CUTIN_MAX_LATERAL_SPEED
        })
        .find(|d| {
            let t_intrude = intrusion_time(d.lateral_speed);
            t_intrude < time_to_red && d.distance_to_ego < d_req
        })
        .map(|_| WarningLevel::Warning)
}

/// Rule 5 (Section 4.6): Short-yellow advisory.
///
/// A yellow signal with less than
/// [`SHORT_YELLOW_THRESHOLD`](constants::SHORT_YELLOW_THRESHOLD)
/// seconds to red leaves insufficient time for a comfortable stop in
/// most traffic and produces a Caution.  Kept after the
/// Warning-level rules in the pipeline so it never masks a more
/// severe warning.
///
/// # Arguments
/// * `light` — The current traffic-light phase.  Only
///   [`LightState::Yellow`] can trigger this rule.
/// * `time_to_red` — Seconds remaining before the signal turns red.
///
/// # Returns
/// `Some(Caution)` if the light is yellow and `time_to_red` is below
/// the threshold; `None` otherwise.
#[must_use]
pub fn rule_yellow(light: LightState, time_to_red: f32) -> Option<WarningLevel> {
    match light {
        LightState::Yellow if time_to_red < constants::SHORT_YELLOW_THRESHOLD => {
            Some(WarningLevel::Caution)
        }
        _ => None,
    }
}

/// Rule 6 (Section 4.6): Worst-case green advisory.
///
/// Engineering heuristic, deliberately excluded from the formal
/// theorems: under the vision-only worst-case interpretation, a
/// green may end at any frame, so this rule advises Caution when a
/// comfortable stop is no longer possible (`d_stop_line >
/// stopping_distance`).
///
/// # Arguments
/// * `light` — The current traffic-light phase.  Only
///   [`LightState::Green`] can trigger this rule.
/// * `ego` — Instantaneous ego state (speed and distance to the
///   stop line).
///
/// # Returns
/// `Some(Caution)` if the light is green and the ego is beyond the
/// comfortable-stop envelope; `None` otherwise.
#[must_use]
pub fn rule_stale(light: LightState, ego: &EgoState) -> Option<WarningLevel> {
    match light {
        LightState::Green if ego.distance_to_stop_line > stopping_distance(ego.speed) => {
            Some(WarningLevel::Caution)
        }
        _ => None,
    }
}

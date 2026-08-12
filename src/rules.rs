//! Decision rules. Each function implements one criterion of
//! Section 4. All functions are pure and return `Option` to
//! facilitate composition.

use crate::algebra::*;
use crate::models::*;

/// Rule 1 (Section 4.6): Light state rule.
/// Red implies Critical; short yellow implies Caution.
#[must_use]
pub fn rule_light(light: LightState, time_to_red: f32) -> Option<WarningLevel> {
    match light {
        LightState::Red => Some(WarningLevel::Critical),
        LightState::Yellow if time_to_red < 2.5 => Some(WarningLevel::Caution),
        _ => None,
    }
}

/// Rule 2 (Theorem 4): Dilemma zone rule.
/// The core stopping-clearance conjunction.
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
/// Checks if the leader is stopped in the box or if following it
/// causes a clearance failure.
#[must_use]
pub fn rule_lead(ego_speed: f32, lead: &LeadVehicle, time_to_red: f32) -> Option<WarningLevel> {
    let d_req = stopping_distance(ego_speed);

    // Sub-rule 3a: leader is stopped and already inside the intersection.
    if lead.speed < 1.0 && lead.distance < d_req && lead.is_in_intersection {
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
/// Detects adjacent vehicles with turn signals that will intrude
/// before the light changes.
#[must_use]
pub fn rule_cutin(detections: &[Detection], ego_speed: f32, time_to_red: f32) -> Option<WarningLevel> {
    let d_req = stopping_distance(ego_speed);

    detections
        .iter()
        .filter(|d| d.is_vehicle() && d.turn_signal_active && d.lane != LanePosition::Same)
        .find(|d| {
            let t_intrude = intrusion_time(d.lateral_speed);
            t_intrude < time_to_red && d.distance_to_ego < d_req
        })
        .map(|_| WarningLevel::Warning)
}

/// Rule 5 (Section 4.6): Worst-case green advisory.
/// Engineering heuristic, deliberately excluded from the formal theorems:
/// under the vision-only worst-case interpretation, a green may end at any
/// frame, so advise Caution when a comfortable stop is no longer possible.
#[must_use]
pub fn rule_stale(light: LightState, ego: &EgoState) -> Option<WarningLevel> {
    match light {
        LightState::Green if ego.distance_to_stop_line > stopping_distance(ego.speed) => {
            Some(WarningLevel::Caution)
        }
        _ => None,
    }
}

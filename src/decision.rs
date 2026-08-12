//! Decision engine pipeline. Composes five rules.
//! Adding a new rule requires inserting it into the vector.

use crate::algebra::constants;
use crate::models::*;
use crate::rules::*;

/// Evaluates the entire traffic scene and returns the highest-priority
/// warning.
///
/// # Pipeline design
/// Rules are ordered by descending severity: the dilemma zone (Critical)
/// and red light (Critical) precede the lead and cut-in rules (Warning),
/// which precede the stale-green heuristic (Caution). The dilemma rule
/// runs before the yellow advisory so a Critical state is never masked
/// by a Caution. `find_map` returns the first rule that fires;
/// `unwrap_or(Safe)` covers the no-warning case.
#[must_use]
pub fn evaluate_safety(
    ego: &EgoState,
    detections: &[Detection],
    light: LightState,
    time_to_red: f32,
) -> WarningLevel {
    // Extract the lead vehicle from detections.
    let lead_opt = detections
        .iter()
        .find(|d| d.is_vehicle() && d.lane == LanePosition::Same)
        .map(|d| LeadVehicle {
            distance: d.distance_to_ego,
            speed: d.speed,
            is_in_intersection: d.distance_to_ego < constants::INTERSECTION_LENGTH,
        });

    // Build the severity-ordered pipeline.
    let rules: Vec<Box<dyn Fn() -> Option<WarningLevel>>> = vec![
        Box::new(|| rule_dilemma(ego, time_to_red)),
        Box::new(|| rule_light(light, time_to_red)),
        Box::new(|| lead_opt.and_then(|l| rule_lead(ego.speed, &l, time_to_red))),
        Box::new(|| rule_cutin(detections, ego.speed, time_to_red)),
        Box::new(|| rule_stale(light, ego)),
    ];

    // Execute the pipeline; the first matching rule determines the level.
    // Bind to a local so the temporary closures drop before the borrows
    // captured from the enclosing scope are released.
    let level = rules
        .into_iter()
        .find_map(|rule| rule())
        .unwrap_or(WarningLevel::Safe);
    level
}

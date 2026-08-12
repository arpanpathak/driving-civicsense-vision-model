//! Decision engine pipeline. Composes the six rules of
//! [`crate::rules`] into a severity-ordered priority chain.
//!
//! Adding a new rule requires:
//! 1. Defining the rule function in [`crate::rules`],
//! 2. Inserting it at the correct severity position in the
//!    `rules` vector of [`evaluate_safety`],
//! 3. Adding a test vector to the canonical scenarios table in
//!    `tests/verification.rs`.

use crate::algebra::constants;
use crate::models::*;
use crate::rules::*;

/// Evaluates the entire traffic scene and returns the
/// highest-priority warning level.
///
/// # Pipeline design
///
/// Rules are ordered strictly by descending severity:
///
/// | Priority | Rule | Severity |
/// |----------|------|----------|
/// | 1st | `rule_red` | Critical |
/// | 2nd | `rule_dilemma` | Critical |
/// | 3rd | `rule_lead` | Critical / Warning |
/// | 4th | `rule_cutin` | Warning |
/// | 5th | `rule_yellow` | Caution |
/// | 6th | `rule_stale` | Caution |
///
/// `find_map` returns the level of the first rule that fires;
/// `unwrap_or(Safe)` covers the no-warning case.  Because the
/// vector is ordered by severity, the first match is always the
/// most severe applicable warning.
///
/// # Complexity
///
/// O(n) per frame where n is the number of detections (only
/// `rule_cutin` iterates over the slice).  In practice n ≤ 12,
/// so the decision cost is negligible relative to the perception
/// stage.
///
/// # Arguments
///
/// * `ego` — Instantaneous state of the ego vehicle (speed and
///   distance to the stop line, assumed synchronised to the same
///   frame timestamp).  See [`EgoState`].
/// * `detections` — Tracked objects from the current frame, each
///   carrying kinematic state, lane assignment, turn-signal status,
///   and track age.  See [`Detection`].
/// * `light` — The current traffic-signal phase at the intersection
///   (Red, Yellow, Green, or Unknown).  See [`LightState`].
/// * `time_to_red` — Seconds remaining before the signal turns red
///   (from V2I/SPaT or a vision-based countdown detector).  Under
///   the vision-only worst-case interpretation a green is treated
///   as potentially ending at any frame; the caller sets
///   `time_to_red` to the frame duration in that mode.
///
/// # Returns
/// The highest-priority [`WarningLevel`] among all rules that fire,
/// or [`WarningLevel::Safe`] if no rule fires.
///
/// # Note on `let_and_return`
///
/// The result is deliberately bound to a local before being
/// returned: the boxed closures borrow from `ego`, `light`,
/// `time_to_red`, and `lead_opt`, and binding the tail expression
/// forces those temporaries to drop before the enclosing scope ends
/// (required by the borrow checker).
#[must_use]
#[allow(clippy::let_and_return)]
pub fn evaluate_safety(
    ego: &EgoState,
    detections: &[Detection],
    light: LightState,
    time_to_red: f32,
) -> WarningLevel {
    // Extract the closest lead vehicle in the same lane.
    let lead_opt = detections
        .iter()
        .find(|d| d.is_vehicle() && d.lane == LanePosition::Same)
        .map(|d| LeadVehicle {
            distance: d.distance_to_ego,
            speed: d.speed,
            is_in_intersection: d.distance_to_ego < constants::INTERSECTION_LENGTH,
        });

    // Build the severity-ordered pipeline.  Each rule returns at
    // most one level; the order guarantees the first match is the
    // most severe.
    let rules: Vec<Box<dyn Fn() -> Option<WarningLevel>>> = vec![
        Box::new(|| rule_red(light)),
        Box::new(|| rule_dilemma(ego, time_to_red)),
        Box::new(|| lead_opt.and_then(|l| rule_lead(ego.speed, &l, time_to_red))),
        Box::new(|| rule_cutin(detections, ego.speed, time_to_red)),
        Box::new(|| rule_yellow(light, time_to_red)),
        Box::new(|| rule_stale(light, ego)),
    ];

    // Execute the pipeline; the first matching rule determines the
    // level.  Bind to a local so the temporary closures drop before
    // the borrows captured from the enclosing scope are released.
    let level = rules
        .into_iter()
        .find_map(|rule| rule())
        .unwrap_or(WarningLevel::Safe);
    level
}

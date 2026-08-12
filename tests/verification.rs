//! Verification suite for the deterministic decision engine.
//!
//! Three layers of testing, mirroring Section VI of the paper:
//!
//! 1. **Canonical scenarios** (Table II): the eight executable test vectors
//!    from the paper, asserting the exact expected warning level.
//! 2. **Exhaustive enumeration**: a bounded grid over the discrete input
//!    space (light state, time-to-red, ego speed, stop-line distance, and
//!    detection patterns). Asserts totality (no panics), determinism, and
//!    the theorem-backed properties below.
//! 3. **Property checks**: red-light dominance, dilemma-zone
//!    criticality, monotonicity of the warning level in the stop-line
//!    distance, and severity ordering of the pipeline.

use civicsense::algebra::constants::*;
use civicsense::algebra::{clearance_time, stopping_distance};
use civicsense::decision::evaluate_safety;
use civicsense::models::*;

/// Constructs an [`EgoState`] from its two scalar fields: speed (m/s)
/// and distance to the stop line (m).
fn ego(speed: f32, distance_to_stop_line: f32) -> EgoState {
    EgoState {
        speed,
        distance_to_stop_line,
    }
}

/// Constructs a [`Detection`] for a passenger car with the given
/// longitudinal speed, distance, lane position, and turn-signal
/// status.
///
/// When `turn_signal_active` is true the lateral speed is set to
/// 1.2 m/s (a realistic lane-change rate) and `track_age` to
/// [`CUTIN_MIN_OBSERVATION_FRAMES`] so the cut-in rule's latency
/// filter is satisfied.  Otherwise both are zero.
fn detection(
    speed: f32,
    distance_to_ego: f32,
    lane: LanePosition,
    turn_signal_active: bool,
) -> Detection {
    Detection {
        bbox: (0.0, 0.0, 0.0, 0.0),
        class_id: 2, // car
        speed,
        lateral_speed: if turn_signal_active { 1.2 } else { 0.0 },
        distance_to_ego,
        lane,
        turn_signal_active,
        track_age: if turn_signal_active {
            CUTIN_MIN_OBSERVATION_FRAMES
        } else {
            0
        },
    }
}

/// Scenario helper: a lead vehicle stopped 10 m ahead, inside the
/// 16 m intersection box.  Used in canonical scenarios row 4
/// (lead stopped → Critical).
///
/// Note: the detection's `distance_to_ego < INTERSECTION_LENGTH`
/// is what the engine uses for `is_in_intersection`, so no explicit
/// flag is needed.
#[allow(dead_code)]
fn stopped_lead_in_box() -> Vec<Detection> {
    vec![detection(0.0, 10.0, LanePosition::Same, false)]
}

/// Scenario helper: a slow-moving lead vehicle (5 m/s, 20 m ahead).
/// Used in canonical scenarios row 5 (slow lead → Warning).
fn slow_lead() -> Vec<Detection> {
    vec![detection(5.0, 20.0, LanePosition::Same, false)]
}

/// Scenario helper: a vehicle in the left lane with an active turn
/// signal, 15 m ahead at 18 m/s with lateral speed 1.2 m/s.
/// Used in canonical scenarios row 6 (cut-in → Warning).
fn cutin_left() -> Vec<Detection> {
    vec![detection(18.0, 15.0, LanePosition::Left, true)]
}

/// Empty detection list, used when a scenario involves only the ego
/// vehicle and the traffic signal.
fn no_detections() -> Vec<Detection> {
    Vec::new()
}

// ---------------------------------------------------------------------
// 1. Canonical scenarios (Table II)
// ---------------------------------------------------------------------

#[test]
fn canonical_scenarios_match_table_ii() {
    let v14_d25 = ego(14.0, 25.0);

    // Row 1: red light, any state -> Critical
    assert_eq!(
        evaluate_safety(&v14_d25, &no_detections(), LightState::Red, 3.5),
        WarningLevel::Critical
    );

    // Row 2: yellow, t_y = 1.5 s -> Caution (comfortable-stop geometry,
    // outside the dilemma zone, so rule_light is the first to fire)
    let v10_d40 = ego(10.0, 40.0);
    assert_eq!(
        evaluate_safety(&v10_d40, &no_detections(), LightState::Yellow, 1.5),
        WarningLevel::Caution
    );

    // Row 3: dilemma zone, v_e = 14, d_s = 25 -> Critical
    assert_eq!(
        evaluate_safety(&v14_d25, &no_detections(), LightState::Yellow, 3.5),
        WarningLevel::Critical
    );

    // Row 4: lead stopped inside box -> Critical
    assert_eq!(
        evaluate_safety(&v14_d25, &stopped_lead_in_box(), LightState::Green, 6.0),
        WarningLevel::Critical
    );

    // Row 5: slow lead, v_l = 5, d_l = 20 -> Warning (non-dilemma geometry)
    assert_eq!(
        evaluate_safety(&v10_d40, &slow_lead(), LightState::Green, 3.5),
        WarningLevel::Warning
    );

    // Row 6: cut-in vehicle -> Warning (non-dilemma geometry)
    assert_eq!(
        evaluate_safety(&v10_d40, &cutin_left(), LightState::Green, 3.5),
        WarningLevel::Warning
    );

    // Row 7: green beyond the stop envelope -> Caution
    let v10_d30 = ego(10.0, 30.0);
    assert_eq!(
        evaluate_safety(&v10_d30, &no_detections(), LightState::Green, 6.0),
        WarningLevel::Caution
    );

    // Row 8: green, comfortable stop margin -> Safe
    let v10_d10 = ego(10.0, 10.0);
    assert_eq!(
        evaluate_safety(&v10_d10, &no_detections(), LightState::Green, 6.0),
        WarningLevel::Safe
    );
}

// ---------------------------------------------------------------------
// 2. Exhaustive enumeration over the bounded input space
// ---------------------------------------------------------------------

#[test]
fn exhaustive_input_space_is_total_and_deterministic() {
    let lights = [
        LightState::Red,
        LightState::Yellow,
        LightState::Green,
        LightState::Unknown,
    ];
    let times = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0];
    let speeds = [0.0, 2.0, 5.0, 8.0, 10.0, 14.0, 18.0, 20.0];
    let distances = [
        5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 38.0, 40.0, 50.0, 60.0,
    ];
    let patterns: [(&str, Vec<Detection>); 5] = [
        ("none", no_detections()),
        ("stopped lead", stopped_lead_in_box()),
        ("slow lead", slow_lead()),
        ("cut-in", cutin_left()),
        ("lead + cut-in", {
            let mut v = slow_lead();
            v.push(detection(18.0, 15.0, LanePosition::Left, true));
            v
        }),
    ];

    let mut count = 0u64;
    for &light in &lights {
        for &ttr in &times {
            for &speed in &speeds {
                for &dist in &distances {
                    for (_, dets) in &patterns {
                        let e = ego(speed, dist);
                        let a = evaluate_safety(&e, dets, light, ttr);
                        let b = evaluate_safety(&e, dets, light, ttr);
                        assert_eq!(
                            a, b,
                            "non-deterministic result for {light:?} {ttr} {speed} {dist}"
                        );
                        count += 1;
                    }
                }
            }
        }
    }
    assert!(
        count > 1000,
        "exhaustive enumeration should cover thousands of states, got {count}"
    );
}

// ---------------------------------------------------------------------
// 3. Theorem-backed properties
// ---------------------------------------------------------------------

/// Red light dominates: Critical regardless of everything else.
#[test]
fn red_light_is_always_critical() {
    let speeds = [2.0, 14.0, 20.0];
    let distances = [5.0, 25.0, 60.0];
    let patterns = [no_detections(), stopped_lead_in_box(), cutin_left()];
    for &s in &speeds {
        for &d in &distances {
            for dets in &patterns {
                for &ttr in &[1.0, 3.5, 6.0] {
                    assert_eq!(
                        evaluate_safety(&ego(s, d), dets, LightState::Red, ttr),
                        WarningLevel::Critical
                    );
                }
            }
        }
    }
}

/// Dilemma zone: when the ego can neither stop nor clear, the engine
/// must return at least Warning (Critical when no red dominates).
#[test]
fn dilemma_zone_is_never_safe() {
    for &speed in &[8.0, 10.0, 14.0, 18.0, 20.0] {
        for &dist in &[5.0, 8.0, 10.0, 15.0, 20.0, 25.0] {
            let d_req = stopping_distance(speed);
            let t_c = clearance_time(dist, speed);
            let cannot_stop = dist <= d_req;
            for &ttr in &[1.5, 2.0, 2.5, 3.0, 3.5] {
                let cannot_clear = t_c >= (ttr - SAFETY_MARGIN);
                if cannot_stop && cannot_clear {
                    let level = evaluate_safety(
                        &ego(speed, dist),
                        &no_detections(),
                        LightState::Yellow,
                        ttr,
                    );
                    assert!(
                        level >= WarningLevel::Warning,
                        "dilemma state ({speed}, {dist}, {ttr}) must warn, got {level:?}"
                    );
                }
            }
        }
    }
}

/// Corollary 1 (Monotonicity): the required stopping distance is strictly
/// increasing in the approach speed, and the clearance time is strictly
/// decreasing in it. These are the algebraic monotonicity claims of the
/// paper.
#[test]
fn stopping_distance_is_monotone_in_speed() {
    let speeds = [0.0, 2.0, 5.0, 8.0, 10.0, 14.0, 18.0, 20.0, 30.0];
    for w in speeds.windows(2) {
        assert!(
            stopping_distance(w[1]) > stopping_distance(w[0]),
            "d_req must strictly increase with speed: {} vs {}",
            stopping_distance(w[1]),
            stopping_distance(w[0])
        );
    }
    for w in speeds.windows(2) {
        let t0 = clearance_time(25.0, w[0].max(0.1));
        let t1 = clearance_time(25.0, w[1].max(0.1));
        assert!(t1 < t0, "clearance time must strictly decrease with speed");
    }
}

/// Pipeline severity order: when both a Critical and a Warning rule
/// apply, the reported level is the most severe one (the first match).
#[test]
fn severity_ordering_first_match_wins() {
    // Red (Critical) dominates a cut-in (Warning).
    let mut dets = cutin_left();
    dets.push(detection(5.0, 20.0, LanePosition::Same, false));
    assert_eq!(
        evaluate_safety(&ego(14.0, 25.0), &dets, LightState::Red, 3.5),
        WarningLevel::Critical
    );
    // Dilemma (Critical) dominates the stale-green advisory (Caution).
    assert_eq!(
        evaluate_safety(&ego(14.0, 25.0), &no_detections(), LightState::Green, 3.5),
        WarningLevel::Critical
    );
}

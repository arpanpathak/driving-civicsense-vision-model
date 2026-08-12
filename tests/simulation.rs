//! Monte Carlo simulation of the decision engine.
//!
//! Generates 10,000 random intersection approaches, computes the
//! ground-truth risk level **independently from the theorem conditions**
//! (stopping distance, clearance time, lead constraints, intrusion time,
//! light rules), and compares it with the pipeline output. Reports a
//! confusion matrix and asserts exact agreement.
//!
//! The oracle does not call any rule function; it re-derives the risk from
//! the kinematic conditions of Section IV, so this test validates that the
//! rule pipeline (including lead extraction and severity ordering) actually
//! implements the theorems.

use civicsense::algebra::constants::*;
use civicsense::algebra::{clearance_time, intrusion_time, stopping_distance};
use civicsense::decision::evaluate_safety;
use civicsense::models::*;

/// Deterministic xorshift64 RNG, so the simulation is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn f(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() as f32 / u64::MAX as f32) * (hi - lo)
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[(self.next() % items.len() as u64) as usize]
    }
}

/// Ground-truth risk from the theorem conditions of Section IV,
/// deliberately computed without calling the rule functions.
fn ground_truth(
    ego: &EgoState,
    dets: &[Detection],
    light: LightState,
    time_to_red: f32,
) -> WarningLevel {
    let d_req = stopping_distance(ego.speed);
    let cannot_stop = ego.distance_to_stop_line <= d_req;
    let cannot_clear =
        clearance_time(ego.distance_to_stop_line, ego.speed) >= (time_to_red - SAFETY_MARGIN);
    let dilemma = cannot_stop && cannot_clear;

    let lead = dets
        .iter()
        .find(|d| d.is_vehicle() && d.lane == LanePosition::Same);
    let lead_blocked = lead.is_some_and(|l| {
        l.speed < 1.0 && l.distance_to_ego < d_req && l.distance_to_ego < INTERSECTION_LENGTH
    });
    let lead_follow = lead.is_some_and(|l| {
        clearance_time(l.distance_to_ego, l.speed) >= (time_to_red - SAFETY_MARGIN)
            && l.distance_to_ego < d_req
    });
    let cutin = dets.iter().any(|d| {
        d.is_vehicle()
            && d.turn_signal_active
            && d.lane != LanePosition::Same
            && intrusion_time(d.lateral_speed) < time_to_red
            && d.distance_to_ego < d_req
    });

    let short_yellow = light == LightState::Yellow && time_to_red < 2.5;
    let stale = light == LightState::Green && ego.distance_to_stop_line > d_req;

    if light == LightState::Red || dilemma || lead_blocked {
        WarningLevel::Critical
    } else if lead_follow || cutin {
        WarningLevel::Warning
    } else if short_yellow || stale {
        WarningLevel::Caution
    } else {
        WarningLevel::Safe
    }
}

fn random_detections(rng: &mut Rng) -> Vec<Detection> {
    let n = rng.next() % 4; // 0..=3 vehicles
    (0..n)
        .map(|_| {
            let lane = rng.pick(&[LanePosition::Same, LanePosition::Left, LanePosition::Right]);
            let signal = rng.f(0.0, 1.0) < 0.25;
            Detection {
                bbox: (0.0, 0.0, 0.0, 0.0),
                class_id: 2, // car
                speed: rng.f(0.0, 20.0),
                lateral_speed: if signal { rng.f(0.5, 3.0) } else { 0.0 },
                distance_to_ego: rng.f(2.0, 60.0),
                lane,
                turn_signal_active: signal,
            }
        })
        .collect()
}

#[test]
fn monte_carlo_10000_scenes_match_theorem_ground_truth() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let lights = [
        LightState::Red,
        LightState::Yellow,
        LightState::Green,
        LightState::Unknown,
    ];

    let mut confusion = [[0u64; 4]; 4]; // rows: pipeline, cols: ground truth
    let mut mismatches = Vec::new();
    let mut counts = [0u64; 4]; // Safe, Caution, Warning, Critical

    const N: u64 = 10_000;
    for _ in 0..N {
        let light = rng.pick(&lights);
        let time_to_red = rng.f(1.0, 12.0);
        let ego = EgoState {
            speed: rng.f(0.0, 25.0),
            distance_to_stop_line: rng.f(0.0, 80.0),
        };
        let dets = random_detections(&mut rng);

        let truth = ground_truth(&ego, &dets, light, time_to_red);
        let out = evaluate_safety(&ego, &dets, light, time_to_red);

        let ti = truth as usize;
        let oi = out as usize;
        confusion[oi][ti] += 1;
        counts[oi] += 1;

        if out != truth {
            mismatches.push((ego, dets, light, time_to_red, truth, out));
        }
    }

    // Report the distribution.
    let names = ["Safe", "Caution", "Warning", "Critical"];
    println!("distribution over {N} random approaches:");
    for (i, n) in names.iter().enumerate() {
        println!("  {n:>9}: {:5.2}%", 100.0 * counts[i] as f64 / N as f64);
    }
    println!("confusion matrix (rows = pipeline, cols = ground truth):");
    print!("          ");
    for n in names {
        print!("{n:>9}");
    }
    println!();
    for (i, n) in names.iter().enumerate() {
        print!("{n:>9}: ");
        for value in confusion[i] {
            print!("{value:>9}");
        }
        println!();
    }

    assert!(
        mismatches.is_empty(),
        "{} mismatches vs theorem ground truth; first: {:?}",
        mismatches.len(),
        mismatches.first()
    );
}

"""Run fixed-time or crossing-specific pedestrian-actuated SUMO control."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

if __package__:
    from .build_network import GENERATED, build
    from .run_simulation import _import_traci, _sumo_binary
else:
    from build_network import GENERATED, build
    from run_simulation import _import_traci, _sumo_binary


MIN_GREEN = 10
SIMULATION_END = 600
WARMUP = 60
PEDESTRIAN_INTERVAL = 20
PEDESTRIAN_START = 10

PEDESTRIAN_ROUTES = (
    ("J0", "N0_J0", ":J0_c0", "J0_S0"),
    ("J0", "S0_J0", ":J0_c2", "J0_N0"),
    ("J0", "W0_J0", ":J0_c3", "J0_J1"),
    ("J0", "J0_J1", ":J0_c1", "J0_W0"),
    ("J1", "N1_J1", ":J1_c0", "J1_S1"),
    ("J1", "S1_J1", ":J1_c2", "J1_N1"),
    ("J1", "J0_J1", ":J1_c3", "J1_J2"),
    ("J1", "J1_J2", ":J1_c1", "J1_J0"),
    ("J2", "N2_J2", ":J2_c0", "J2_S2"),
    ("J2", "S2_J2", ":J2_c2", "J2_N2"),
    ("J2", "J1_J2", ":J2_c3", "J2_E2"),
    ("J2", "E2_J2", ":J2_c1", "J2_J1"),
)


def _crossing_phase_map(traci, tls_id: str) -> dict[str, int]:
    links = traci.trafficlight.getControlledLinks(tls_id)
    phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
    mapping = {}
    for index, link_group in enumerate(links):
        if not link_group:
            continue
        destination = link_group[0][1]
        if not destination.startswith(f":{tls_id}_c"):
            continue
        crossing = destination.rsplit("_", 1)[0]
        candidates = [
            phase_index
            for phase_index, phase in enumerate(phases)
            if phase.state[index].lower() == "g"
        ]
        if candidates:
            mapping[crossing] = candidates[0]
    return mapping


def _vehicle_routes() -> dict[str, tuple[str, ...]]:
    return {
        "corridor-east": ("W0_J0", "J0_J1", "J1_J2", "J2_E2"),
        "corridor-west": ("E2_J2", "J2_J1", "J1_J0", "J0_W0"),
        "north-south-0": ("N0_J0", "J0_S0"),
        "south-north-0": ("S0_J0", "J0_N0"),
        "north-south-1": ("N1_J1", "J1_S1"),
        "south-north-1": ("S1_J1", "J1_N1"),
        "north-south-2": ("N2_J2", "J2_S2"),
        "south-north-2": ("S2_J2", "J2_N2"),
    }


def run(
    seed: int,
    duration: int,
    output: Path,
    rebuild: bool,
    controller: str = "pedestrian_actuated",
    green_duration: int = 37,
    pedestrian_speed: float = 1.4,
) -> dict[str, object]:
    if pedestrian_speed <= 0:
        raise ValueError("pedestrian_speed must be positive")
    traci = _import_traci()
    if rebuild or not (GENERATED / "simulation.sumocfg").exists():
        build(green_duration, pedestrian_speed)
    traci.start(
        [
            _sumo_binary("sumo"),
            "-c",
            str(GENERATED / "simulation.sumocfg"),
            "--seed",
            str(seed),
            "--end",
            str(duration),
            "--step-length",
            "1",
        ]
    )
    vehicle_wait = 0.0
    pedestrian_wait = 0.0
    vehicle_wait_completed = 0.0
    pedestrian_wait_completed = 0.0
    vehicle_arrivals = 0
    pedestrian_arrivals = 0
    vehicle_completed = 0
    pedestrian_completed = 0
    pedestrian_id = 0
    pedestrian_meta: dict[str, tuple[str, str]] = {}
    pedestrian_wait_by_id = defaultdict(float)
    vehicle_wait_by_id = defaultdict(float)
    vehicle_time_loss_by_id = defaultdict(float)
    vehicle_time_loss = 0.0
    vehicle_completed_time_loss = 0.0
    pedestrian_waits_completed: list[float] = []
    wait_by_junction = defaultdict(float)
    wait_by_crossing = defaultdict(float)
    phase_age = {tls_id: 0 for tls_id in traci.trafficlight.getIDList()}
    crossing_phases = {
        tls_id: _crossing_phase_map(traci, tls_id) for tls_id in phase_age
    }

    try:
        while traci.simulation.getTime() < duration:
            now = int(traci.simulation.getTime())
            if now >= PEDESTRIAN_START and now % PEDESTRIAN_INTERVAL == 0:
                for junction, start, crossing, end in PEDESTRIAN_ROUTES:
                    person_id = f"ped-{pedestrian_id}"
                    traci.person.add(person_id, start, 0, now)
                    traci.person.appendWalkingStage(
                        person_id, [start, crossing, end], 0, pedestrian_speed
                    )
                    pedestrian_meta[person_id] = (junction, crossing)
                    pedestrian_id += 1

            for tls_id in phase_age:
                phase_age[tls_id] += 1
                if controller != "pedestrian_actuated":
                    continue
                requests = [
                    crossing
                    for person_id, (junction, crossing) in pedestrian_meta.items()
                    if junction == tls_id
                    and person_id in traci.person.getIDList()
                    and traci.person.getSpeed(person_id) < 0.1
                    and traci.person.getRoadID(person_id) != crossing
                ]
                current_phase = traci.trafficlight.getPhase(tls_id)
                requested_phases = {
                    crossing_phases[tls_id][crossing]
                    for crossing in requests
                    if crossing in crossing_phases[tls_id]
                }
                if (
                    requested_phases
                    and phase_age[tls_id] >= MIN_GREEN
                    and current_phase not in requested_phases
                    and current_phase in (0, 3)
                ):
                    traci.trafficlight.setPhaseDuration(tls_id, 0)
                    phase_age[tls_id] = 0

            traci.simulationStep()
            active_vehicles = traci.vehicle.getIDList()
            active_people = traci.person.getIDList()
            departed = traci.simulation.getDepartedIDList()
            arrived_vehicles = traci.simulation.getArrivedIDList()
            arrived_people = traci.simulation.getArrivedPersonIDList()
            vehicle_arrivals += len(departed)
            vehicle_completed += len(arrived_vehicles)
            pedestrian_completed += len(arrived_people)
            current_vehicle_wait = sum(
                traci.vehicle.getSpeed(vehicle_id) < 0.1
                for vehicle_id in active_vehicles
            )
            vehicle_wait += current_vehicle_wait
            for vehicle_id in active_vehicles:
                stopped = traci.vehicle.getSpeed(vehicle_id) < 0.1
                if stopped:
                    vehicle_wait_by_id[vehicle_id] += 1
                time_loss = traci.vehicle.getTimeLoss(vehicle_id)
                vehicle_time_loss += max(
                    0.0, time_loss - vehicle_time_loss_by_id[vehicle_id]
                )
                vehicle_time_loss_by_id[vehicle_id] = time_loss
            for person_id in active_people:
                stopped = traci.person.getSpeed(person_id) < 0.1
                if stopped:
                    pedestrian_wait += 1
                    pedestrian_wait_by_id[person_id] += 1
                    junction, crossing = pedestrian_meta[person_id]
                    wait_by_junction[junction] += 1
                    wait_by_crossing[crossing] += 1
            for vehicle_id in arrived_vehicles:
                vehicle_wait_completed += vehicle_wait_by_id.pop(vehicle_id, 0)
                vehicle_completed_time_loss += vehicle_time_loss_by_id.pop(
                    vehicle_id, 0.0
                )
            for person_id in arrived_people:
                waiting = pedestrian_wait_by_id.pop(person_id, 0)
                pedestrian_wait_completed += waiting
                pedestrian_waits_completed.append(waiting)
    finally:
        traci.close()

    metrics: dict[str, object] = {
        "controller": controller,
        "seed": seed,
        "duration_seconds": duration,
        "green_duration_seconds": green_duration,
        "pedestrian_speed_meters_per_second": pedestrian_speed,
        "warmup_seconds": WARMUP,
        "vehicle_waiting_seconds": vehicle_wait,
        "pedestrian_waiting_seconds": pedestrian_wait,
        "vehicle_completed_waiting_seconds": vehicle_wait_completed,
        "pedestrian_completed_waiting_seconds": pedestrian_wait_completed,
        "vehicle_average_waiting_seconds": vehicle_wait / max(vehicle_arrivals, 1),
        "pedestrian_average_waiting_seconds": pedestrian_wait / max(pedestrian_id, 1),
        "vehicle_completed_average_waiting_seconds": vehicle_wait_completed
        / max(vehicle_completed, 1),
        "pedestrian_completed_average_waiting_seconds": pedestrian_wait_completed
        / max(pedestrian_completed, 1),
        "vehicle_arrivals": vehicle_arrivals,
        "pedestrian_arrivals": pedestrian_id,
        "vehicle_completed": vehicle_completed,
        "pedestrian_completed": pedestrian_completed,
        "vehicle_unfinished": len(vehicle_wait_by_id),
        "pedestrian_unfinished": len(pedestrian_wait_by_id),
        "vehicle_time_loss_seconds": vehicle_time_loss,
        "vehicle_completed_time_loss_seconds": vehicle_completed_time_loss,
        "pedestrian_p95_waiting_seconds": (
            statistics.quantiles(
                pedestrian_waits_completed
                + list(pedestrian_wait_by_id.values()),
                n=20,
                method="inclusive",
            )[18]
            if len(pedestrian_waits_completed) + len(pedestrian_wait_by_id) > 1
            else (
                pedestrian_waits_completed[0]
                if pedestrian_waits_completed
                else next(iter(pedestrian_wait_by_id.values()), 0.0)
            )
        ),
        "pedestrian_completed_p95_waiting_seconds": (
            statistics.quantiles(
                pedestrian_waits_completed,
                n=20,
                method="inclusive",
            )[18]
            if len(pedestrian_waits_completed) > 1
            else (
                pedestrian_waits_completed[0]
                if pedestrian_waits_completed
                else 0.0
            )
        ),
        "waiting_by_junction": dict(wait_by_junction),
        "waiting_by_crossing": dict(wait_by_crossing),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=SIMULATION_END)
    parser.add_argument("--output", type=Path, default=GENERATED / "baseline.json")
    parser.add_argument("--controller", choices=("pedestrian_actuated", "fixed_time"))
    parser.add_argument("--green-duration", type=int, default=37)
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.seed,
                args.duration,
                args.output,
                not args.no_rebuild,
                args.controller or "pedestrian_actuated",
                args.green_duration,
            ),
            indent=2,
        )
    )

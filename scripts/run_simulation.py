"""Run SUMO with an adaptive traffic-light controller."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

if __package__:
    from .build_network import GENERATED, build
else:
    from build_network import GENERATED, build


MIN_GREEN = 10
MAX_GREEN = 45
DECISION_INTERVAL = 10
QUEUE_EXTENSION_THRESHOLD = 3


def _sumo_home() -> Path | None:
    configured = os.environ.get("SUMO_HOME")
    if configured:
        return Path(configured)
    try:
        import sumo
    except ImportError:
        return None
    return Path(sumo.__file__).resolve().parent


def _import_traci():
    try:
        import traci
    except ImportError as error:
        sumo_home = _sumo_home()
        tools = sumo_home / "tools" if sumo_home else None
        if tools is None or not (tools / "traci").exists():
            raise RuntimeError(
                "TraCI is unavailable. Install dependencies with "
                "'python -m pip install -r requirements.txt', or set SUMO_HOME."
            ) from error
        sys.path.insert(0, str(tools))
        import traci
    return traci


def _sumo_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary:
        return binary
    sumo_home = _sumo_home()
    if sumo_home:
        candidates = [
            sumo_home / "bin" / f"{name}.exe",
            sumo_home / "sumo" / "bin" / f"{name}.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    raise RuntimeError(
        f"{name} was not found. Install dependencies with "
        "'python -m pip install -r requirements.txt', or add SUMO to PATH."
    )


def run(gui: bool, rebuild: bool, delay_ms: int) -> None:
    traci = _import_traci()
    fatal_traci_error = traci.exceptions.FatalTraCIError
    if rebuild or not (GENERATED / "simulation.sumocfg").exists():
        build()
    binary_name = "sumo-gui" if gui else "sumo"
    binary = _sumo_binary(binary_name)

    command = [binary, "-c", str(GENERATED / "simulation.sumocfg")]
    if gui and delay_ms > 0:
        command.extend(["--delay", str(delay_ms)])
    traci.start(command)
    elapsed_green = {tls_id: 0 for tls_id in traci.trafficlight.getIDList()}
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            simulation_time = traci.simulation.getTime()
            try:
                traci.simulationStep()
            except fatal_traci_error:
                if simulation_time >= 3599:
                    break
                raise
            if int(traci.simulation.getTime()) % DECISION_INTERVAL != 0:
                continue
            for tls_id in elapsed_green:
                elapsed_green[tls_id] += DECISION_INTERVAL
                lanes = set(traci.trafficlight.getControlledLanes(tls_id))
                queue = sum(
                    traci.lane.getLastStepHaltingNumber(lane_id)
                    for lane_id in lanes
                    if lane_id
                )
                phase = traci.trafficlight.getPhase(tls_id)
                if (
                    elapsed_green[tls_id] >= MIN_GREEN
                    and (
                        queue < QUEUE_EXTENSION_THRESHOLD
                        or elapsed_green[tls_id] >= MAX_GREEN
                    )
                ):
                    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                    green_phases = [
                        index
                        for index, candidate in enumerate(logic.phases)
                        if "g" in candidate.state.lower()
                    ]
                    if green_phases:
                        next_phase = green_phases[
                            (green_phases.index(phase) + 1) % len(green_phases)
                        ] if phase in green_phases else green_phases[0]
                        traci.trafficlight.setPhase(tls_id, next_phase)
                        elapsed_green[tls_id] = 0
                traci.trafficlight.setPhaseDuration(
                    tls_id, max(1, MIN_GREEN - elapsed_green[tls_id])
                )
    finally:
        try:
            traci.close()
        except fatal_traci_error:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nogui", action="store_true", help="Run without the SUMO GUI.")
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Reuse the generated network instead of invoking netconvert.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=100,
        help="GUI delay in milliseconds per simulation step.",
    )
    arguments = parser.parse_args()
    run(
        gui=not arguments.nogui,
        rebuild=not arguments.no_rebuild,
        delay_ms=arguments.delay,
    )

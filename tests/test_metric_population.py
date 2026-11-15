from types import SimpleNamespace

import pytest

from scripts import run_baseline as baseline
from scripts.train_marl import SignalEnvironment


def test_baseline_counts_moving_unfinished_trips_and_zero_wait_people(tmp_path, monkeypatch):
    import sys

    clock = [20]
    people = []
    traci = SimpleNamespace(
        start=lambda args: None,
        close=lambda: None,
        simulationStep=lambda: clock.__setitem__(0, clock[0] + 1),
        trafficlight=SimpleNamespace(getIDList=lambda: []),
        simulation=SimpleNamespace(
            getTime=lambda: clock[0], getDepartedIDList=lambda: ["car"],
            getArrivedIDList=lambda: [], getArrivedPersonIDList=lambda: [],
        ),
        vehicle=SimpleNamespace(
            getIDList=lambda: ["car"], getSpeed=lambda _: 1.0,
            getTimeLoss=lambda _: 0.0,
        ),
        person=SimpleNamespace(
            add=lambda person, *args: people.append(person),
            appendWalkingStage=lambda *args: None,
            getIDList=lambda: people,
            getSpeed=lambda person: 0.0 if person == "ped-0" else 1.0,
        ),
    )
    monkeypatch.setitem(sys.modules, "libsumo", traci)
    monkeypatch.setattr(baseline, "GENERATED", tmp_path)
    monkeypatch.setattr(baseline, "_sumo_binary", lambda _: "sumo")
    (tmp_path / "simulation.sumocfg").touch()
    metrics = baseline.run(11, 21, tmp_path / "result.json", False, "fixed_time")
    assert metrics["vehicle_unfinished"] == 1
    assert metrics["pedestrian_unfinished"] == 12
    assert metrics["pedestrian_p95_waiting_seconds"] == pytest.approx(0.45)
    assert metrics["warmup_seconds"] == 0


def test_learned_p95_includes_moving_people_with_zero_wait():
    environment = SignalEnvironment.__new__(SignalEnvironment)
    people = [f"ped-{i}" for i in range(12)]
    environment.traci = SimpleNamespace(
        vehicle=SimpleNamespace(getIDList=lambda: []),
        person=SimpleNamespace(getIDList=lambda: people),
    )
    environment.pedestrian_wait_by_id = {people[0]: 1.0}
    environment.pedestrian_meta = {person: ("J0", ":J0_c0") for person in people}
    environment.pedestrian_completed_waits = []
    environment.pedestrian_waits_by_crossing = {}
    environment.waiting_by_crossing = {}
    for field in ("vehicle_arrivals", "vehicle_completed", "pedestrian_arrivals",
                  "pedestrian_completed", "vehicle_waiting", "pedestrian_waiting",
                  "vehicle_time_loss", "vehicle_completed_time_loss",
                  "vehicle_completed_waiting", "pedestrian_completed_waiting"):
        setattr(environment, field, 0)
    metrics = environment.metrics()
    assert metrics["pedestrian_p95_waiting_seconds"] == pytest.approx(0.45)
    assert metrics["pedestrian_p95_by_crossing"][":J0_c0"] == pytest.approx(0.45)


def test_pedestrian_arrivals_follow_seconds_between_policy_decisions(monkeypatch):
    import torch
    from scripts import train_marl

    clock, people, arrivals = [10], [], []
    def add(person, start, position, now):
        people.append(person)
        arrivals.append(now)
    traci = SimpleNamespace(
        simulation=SimpleNamespace(getTime=lambda: clock[0],
            getDepartedIDList=lambda: [], getArrivedIDList=lambda: [],
            getArrivedPersonIDList=lambda: []),
        simulationStep=lambda: clock.__setitem__(0, clock[0]+1),
        trafficlight=SimpleNamespace(getPhase=lambda _: 2),
        vehicle=SimpleNamespace(getIDList=lambda: []),
        person=SimpleNamespace(add=add, appendWalkingStage=lambda *args: None,
            getIDList=lambda: people, getSpeed=lambda _: 1.0),
    )
    monkeypatch.setattr(train_marl, "_import_traci", lambda: traci)
    environment = SignalEnvironment(100, duration=15, pedestrian_interval=12)
    environment.traci = traci
    environment.time = 10
    environment.observe = lambda: torch.zeros(3, train_marl.OBSERVATION_SIZE)
    environment.step(torch.zeros(3, dtype=torch.int64))
    assert arrivals == [12] * 12
    assert environment.pedestrian_arrivals == 12

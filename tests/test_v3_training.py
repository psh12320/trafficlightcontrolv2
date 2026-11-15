from types import SimpleNamespace

import pytest
import torch

from scripts import train_marl as marl


def test_approach_walking_area_detection_occupancy_and_current_urgency():
    # The assigned route crossing can differ from the actual next crossing.
    # Shared corridor start edges must not duplicate one pedestrian's demand.
    persons = {
        "start": ("N0_J0", 0.0, ":J0_w0", 30.0, ":J0_c0"),
        "walking": (":J1_w2", 0.0, ":J1_c2", 150.0, ":J1_c0"),
        "occupied": (":J2_c1", 1.0, ":J2_w0", 0.0, ":J2_c1"),
        "occupied_stopped": (":J2_c1", 0.0, ":J2_w0", 70.0, ":J2_c1"),
        "moving": ("N0_J0", 1.0, ":J0_c0", 0.0, ":J0_c0"),
        "other": (":J1_w2", 0.0, "J1_S1", 200.0, ":J1_c0"),
        "shared": ("J0_J1", 0.0, ":J1_w0", 10.0, ":J1_c3"),
    }
    e = marl.SignalEnvironment.__new__(marl.SignalEnvironment)
    e.pedestrian_meta = {p: ("unused", v[4]) for p, v in persons.items()}
    e.traci = SimpleNamespace(person=SimpleNamespace(
        getIDList=lambda: list(persons), getRoadID=lambda p: persons[p][0],
        getSpeed=lambda p: persons[p][1], getNextEdge=lambda p: persons[p][2],
        getWaitingTime=lambda p: persons[p][3]))
    features = e._pedestrian_features()
    assert features["J0"][0] == pytest.approx([0.1, 0, 0, 0])
    assert features["J0"][1] == pytest.approx([0.25, 0, 0, 0])
    assert features["J1"][0] == pytest.approx([0, 0, 0.1, 0.1])
    assert features["J1"][1] == pytest.approx([0, 0, 1.0, 10 / 120])
    assert features["J2"][2] == pytest.approx([0, 0.4, 0, 0])
    assert sum(e._last_crossing_max_waits) == 190


def test_current_observation_layout_and_vehicle_ablation():
    e = marl.SignalEnvironment.__new__(marl.SignalEnvironment)
    e.traci = SimpleNamespace(
        trafficlight=SimpleNamespace(getControlledLanes=lambda j: [j+str(i) for i in range(4)],
                                     getPhase=lambda j: 3),
        lane=SimpleNamespace(getLastStepHaltingNumber=lambda lane: 2,
                             getLastStepVehicleNumber=lambda lane: 5))
    e.phase_age = {j: 15 for j in marl.AGENTS}
    e._pedestrian_features = lambda: {j: ([0.1]*4, [0.2]*4, [0.3]*4) for j in marl.AGENTS}
    e.observation_mode = "full"
    values = e.observe()
    assert values.shape == (3, 25)
    assert values[0, 4:8].tolist() == pytest.approx([0.2]*4)
    assert values[0, 8:12].tolist() == pytest.approx([0.1]*4)
    assert values[0, 12:16].tolist() == pytest.approx([0.2]*4)
    assert values[0, 16:20].tolist() == pytest.approx([0.3]*4)
    e.observation_mode = "vehicle"
    assert torch.count_nonzero(e.observe()[:, 8:20]) == 0
    e.observation_mode = "pedestrian"
    assert torch.count_nonzero(e.observe()[:, :8]) == 0


def test_v3_urgency_is_unclipped_crossing_specific_and_weighted():
    e = marl.SignalEnvironment.__new__(marl.SignalEnvironment)
    e.vehicle_weight, e.pedestrian_weight, e.switch_penalty = 2.0, 3.0, 8.0
    e._last_crossing_max_waits = [60, 120] + [0]*10
    e.reward_mode = "vehicle_pedestrian"
    assert e._reward(20.8, 20.0, 1) == pytest.approx(-(2 + 3*(1 + 2.5) + 8))
    e.reward_mode = "vehicle"
    assert e._reward(20.8, 20.0, 1) == pytest.approx(-10)


def test_two_action_masks_enforce_minimum_green_and_clearance():
    e = marl.SignalEnvironment.__new__(marl.SignalEnvironment)
    phases = {"J0": 0, "J1": 3, "J2": 2}
    e.traci = SimpleNamespace(trafficlight=SimpleNamespace(getPhase=lambda j: phases[j]))
    e.phase_age = {"J0": 5, "J1": 10, "J2": 50}
    e.action_size = 2
    assert e.action_masks().tolist() == [[True, False], [True, True], [True, False]]
    assert marl.Actor().network[-1].out_features == 2

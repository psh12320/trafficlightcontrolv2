from pathlib import Path
from xml.etree import ElementTree

from scripts.build_network import build, create_edges, create_nodes, create_routes, GENERATED
from scripts.train_marl import _gae
import torch


def test_network_has_three_signalized_junctions_and_twelve_crossings(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.build_network.GENERATED", tmp_path)
    create_nodes()
    root = ElementTree.parse(tmp_path / "network.nod.xml").getroot()
    assert len(root.findall("node[@type='traffic_light']")) == 3
    assert len(root.findall("crossing")) == 12


def test_routes_use_defined_edges(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.build_network.GENERATED", tmp_path)
    create_edges()
    create_routes()
    edge_ids = {edge.get("id") for edge in ElementTree.parse(tmp_path / "network.edg.xml").getroot()}
    route_root = ElementTree.parse(tmp_path / "traffic.rou.xml").getroot()
    for route in route_root.findall("route"):
        assert set(route.get("edges", "").split()).issubset(edge_ids)


def test_signal_timing_sets_green_and_clearance_phases(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.build_network.GENERATED", tmp_path)
    build(green_duration=25)
    phases = ElementTree.parse(tmp_path / "network.net.xml").getroot().find(".//tlLogic").findall("phase")
    assert [int(phase.get("duration")) for phase in phases] == [25, 5, 5, 25, 5, 5]


def test_slow_pedestrian_speed_increases_clearance_time(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.build_network.GENERATED", tmp_path)
    build(green_duration=25, pedestrian_speed=0.8)
    phases = ElementTree.parse(tmp_path / "network.net.xml").getroot().find(".//tlLogic").findall("phase")
    assert [int(phases[index].get("duration")) for index in (2, 5)] == [8, 8]


def test_gae_returns_finite_advantages():
    advantages = _gae(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.5, 0.5, 0.5]),
        torch.tensor(0.0),
        0.99,
        0.95,
    )
    assert advantages.shape == (3,)
    assert torch.isfinite(advantages).all()


def test_observation_size_matches_policy_input():
    from scripts.train_marl import Actor, OBSERVATION_SIZE

    actor = Actor()
    assert actor.network[0].in_features == OBSERVATION_SIZE


def test_supported_action_spaces_have_expected_policy_outputs():
    from scripts.train_marl import Actor

    assert Actor(2).network[-1].out_features == 2
    assert Actor(4).network[-1].out_features == 4


def test_proposal_reward_objectives_are_distinct():
    from scripts.train_marl import SignalEnvironment

    environment = SignalEnvironment.__new__(SignalEnvironment)
    environment.vehicle_weight = 1.0
    environment.pedestrian_weight = 1.0
    environment.switch_penalty = 0.0
    environment.reward_mode = "vehicle"
    vehicle_only = environment._reward(2.0, 3.0, 0)
    environment.reward_mode = "vehicle_pedestrian"
    combined = environment._reward(2.0, 3.0, 0)
    assert vehicle_only == -2.0
    assert combined == -5.0

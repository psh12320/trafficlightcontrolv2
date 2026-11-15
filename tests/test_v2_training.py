import json
from pathlib import Path
from xml.etree import ElementTree

import pytest
import torch

from scripts import build_network, run_marl_sweep, train_marl


def test_proposal_rewards_share_normalization_and_switch_cost():
    environment = train_marl.SignalEnvironment.__new__(train_marl.SignalEnvironment)
    environment.vehicle_weight = 1.0
    environment.pedestrian_weight = 1.0
    environment.switch_penalty = train_marl.DEFAULT_SWITCH_PENALTY
    environment.reward_mode = "vehicle"
    assert environment._reward(20.8, 20.0, 2) == pytest.approx(-7.0)
    environment.reward_mode = "vehicle_pedestrian"
    assert environment._reward(20.8, 20.0, 2) == pytest.approx(-8.0)


def test_network_builds_do_not_touch_shared_generated_files(tmp_path, monkeypatch):
    shared = tmp_path/"shared"
    monkeypatch.setattr(build_network, "GENERATED", shared)
    for directory, period in ((tmp_path/"worker-a", 4), (tmp_path/"worker-b", 8)):
        build_network.build(vehicle_period=period, output_dir=directory)
        routes = ElementTree.parse(directory/"traffic.rou.xml").getroot()
        assert {flow.get("period") for flow in routes.findall("flow")} == {str(period)}
        assert (directory/"simulation.sumocfg").is_file()
    assert not shared.exists()


def test_randomized_training_is_isolated_reproducible_and_validation_is_fixed(tmp_path, monkeypatch):
    built, evaluation_demands, configurations = [], [], []
    def fake_build(green, speed, period, output_dir=None):
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir/"simulation.sumocfg").touch()
        built.append((output_dir, period))
    class FakeEnvironment:
        def __init__(self, *args, frame_stack=1):
            assert frame_stack == 1
            self.steps = 0
            configurations.append(Path(args[-1]))
        def reset(self):
            return torch.zeros(3, train_marl.OBSERVATION_SIZE)
        def action_masks(self):
            return torch.ones(3, 4, dtype=torch.bool)
        def step(self, actions):
            self.steps += 1
            return torch.full((3, train_marl.OBSERVATION_SIZE), self.steps/10), -1.0, self.steps==3
        def close(self):
            pass
    def fake_evaluate(checkpoint, seeds, duration, speed, interval, period, skip_build):
        evaluation_demands.append((period, interval, skip_build))
        return [{"seed": seed, "vehicle_time_loss_seconds": 1.0} for seed in seeds]
    monkeypatch.setattr(train_marl, "build", fake_build)
    monkeypatch.setattr(train_marl, "SignalEnvironment", FakeEnvironment)
    monkeypatch.setattr(run_marl_sweep, "evaluate", fake_evaluate)
    schedules = []
    for name in ("run-a", "run-b"):
        output = tmp_path/name
        metrics = run_marl_sweep.run_configuration("ippo", "vehicle_pedestrian", 4,
            8, 15, 100, [21,22,23], 30, 1.4, 4, 20, 6, output,
            randomize_demand=True)
        config_root = metrics.parent
        schedules.append([json.loads(line) for line in (config_root/"demand-schedule.jsonl").read_text().splitlines()])
        saved = torch.load(config_root/"final.pt", weights_only=True)
        assert saved["metadata"]["randomize_demand"] is True
        assert saved["metadata"]["frame_stack"] == 1
        assert saved["metadata"]["bc_episodes"] == 0
        assert saved["metadata"]["switch_penalty"] == 3.0
        assert saved["metadata"]["vehicle_period"] == 6
        assert saved["metadata"]["pedestrian_interval"] == 20
    assert schedules[0] == schedules[1]
    assert {row["vehicle_period"] for row in schedules[0]} == {4,6,8}
    assert {row["pedestrian_interval"] for row in schedules[0]} == {12,20,30}
    assert set(evaluation_demands) == {(6,20,True)}
    assert all(not path.exists() for path in configurations)
    assert len({path for path, period in built}) == 6

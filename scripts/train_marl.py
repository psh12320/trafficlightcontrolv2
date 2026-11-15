"""Train canonical clipped-PPO IPPO or MAPPO signal controllers."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import torch
from torch import nn
from torch.distributions import Categorical

if __package__:
    from .build_network import GENERATED, build
    from .run_baseline import PEDESTRIAN_ROUTES, _crossing_phase_map, _p95
    from .run_simulation import _import_traci, _sumo_binary
else:
    from build_network import GENERATED, build
    from run_baseline import PEDESTRIAN_ROUTES, _crossing_phase_map, _p95
    from run_simulation import _import_traci, _sumo_binary


AGENTS = ("J0", "J1", "J2")
OBSERVATION_SIZE = 25
ACTION_SIZE = 2
ACTION_SIZES = (2, 4)
DECISION_INTERVAL = 5
TRAINING_HORIZON = 120
MIN_GREEN = 10
MAX_GREEN = 45
VEHICLE_SCALE = 20.8
PEDESTRIAN_SCALE = 20.0
DEFAULT_SWITCH_PENALTY = 3.0
URGENCY_COEFFICIENT = 0.5
URGENCY_WAIT_SCALE = 60.0
CROSSING_APPROACHES = {crossing: start for _, start, crossing, _ in PEDESTRIAN_ROUTES}
VEHICLE_PERIOD_CHOICES = (4, 6, 8)
PEDESTRIAN_INTERVAL_CHOICES = (12, 20, 30)


class SignalEnvironment:
    """TraCI environment with safe signal timing actions."""

    def __init__(
        self,
        seed: int,
        duration: int = TRAINING_HORIZON,
        reward_mode: str = "raw",
        vehicle_weight: float = 1.0,
        pedestrian_weight: float = 1.0,
        switch_penalty: float = DEFAULT_SWITCH_PENALTY,
        observation_mode: str = "full",
        action_size: int = ACTION_SIZE,
        green_duration: int = 37,
        pedestrian_speed: float = 1.4,
        pedestrian_interval: int = 20,
        configuration: Path | None = None,
        frame_stack: int = 1,
    ):
        try:
            import libsumo as traci
        except ImportError:
            traci = _import_traci()
        self.traci = traci
        self.seed = seed
        self.configuration = configuration or GENERATED / "simulation.sumocfg"
        if frame_stack not in (1, 3):
            raise ValueError("frame_stack must be 1 or 3")
        self.frame_stack = frame_stack
        self._frame_history = []
        self._frame_time = None
        self.duration = duration
        self.reward_mode = reward_mode
        self.vehicle_weight = vehicle_weight
        self.pedestrian_weight = pedestrian_weight
        self.switch_penalty = switch_penalty
        self.observation_mode = observation_mode
        if action_size not in (2, 4):
            raise ValueError("action_size must be 2 or 4")
        self.action_size = action_size
        self.green_duration = green_duration
        if pedestrian_speed <= 0:
            raise ValueError("pedestrian_speed must be positive")
        self.pedestrian_speed = pedestrian_speed
        if pedestrian_interval <= 0:
            raise ValueError("pedestrian_interval must be positive")
        self.pedestrian_interval = pedestrian_interval
        self.time = 0
        self.pedestrian_id = 0
        self.pedestrian_arrivals = 0
        self.vehicle_waiting = 0.0
        self.pedestrian_waiting = 0.0
        self.vehicle_arrivals = 0
        self.vehicle_completed = 0
        self.vehicle_wait_by_id: dict[str, float] = {}
        self.vehicle_time_loss_by_id: dict[str, float] = {}
        self.vehicle_time_loss = 0.0
        self.vehicle_completed_time_loss = 0.0
        self.vehicle_completed_waiting = 0.0
        self.pedestrian_meta: dict[str, tuple[str, str]] = {}
        self.pedestrian_wait_by_id: dict[str, float] = {}
        self.pedestrian_completed = 0
        self.pedestrian_completed_waiting = 0.0
        self.pedestrian_completed_waits: list[float] = []
        self.pedestrian_waits_by_crossing: dict[str, list[float]] = {}
        self.waiting_by_crossing: dict[str, float] = {}
        self.phase_age = {agent: 0 for agent in AGENTS}
        self.last_phase = {agent: None for agent in AGENTS}
        self.crossing_phases: dict[str, dict[str, int]] = {}

    def reset(self) -> torch.Tensor:
        self.traci.start(
            [
                _sumo_binary("sumo"),
                "-c",
                str(self.configuration),
                "--seed",
                str(self.seed),
                "--end",
                str(self.duration),
                "--step-length",
                "1",
                "--no-step-log",
                "true",
                "--no-warnings",
                "true",
            ]
        )
        self.time = 0
        self.pedestrian_id = 0
        self.pedestrian_arrivals = 0
        self.vehicle_waiting = 0.0
        self.pedestrian_waiting = 0.0
        self.vehicle_arrivals = 0
        self.vehicle_completed = 0
        self.vehicle_wait_by_id = {}
        self.vehicle_time_loss_by_id = {}
        self.vehicle_time_loss = 0.0
        self.vehicle_completed_time_loss = 0.0
        self.vehicle_completed_waiting = 0.0
        self.pedestrian_meta = {}
        self.pedestrian_wait_by_id = {}
        self.pedestrian_completed = 0
        self.pedestrian_completed_waiting = 0.0
        self.pedestrian_completed_waits = []
        self.pedestrian_waits_by_crossing = {}
        self.waiting_by_crossing = {}
        self.phase_age = {agent: 0 for agent in AGENTS}
        self.last_phase = {
            agent: self.traci.trafficlight.getPhase(agent) for agent in AGENTS
        }
        self.crossing_phases = {
            agent: _crossing_phase_map(self.traci, agent) for agent in AGENTS
        }
        self._frame_history = []
        self._frame_time = None
        return self.observe()

    def _pedestrian_features(self) -> dict[str, tuple[list[float], list[float], list[float]]]:
        counts = dict.fromkeys(CROSSING_APPROACHES, 0)
        oldest = dict.fromkeys(CROSSING_APPROACHES, 0.0)
        occupancies = dict.fromkeys(CROSSING_APPROACHES, 0)
        for person in self.traci.person.getIDList():
            road = self.traci.person.getRoadID(person)
            if road in occupancies:
                occupancies[road] += 1
                continue
            if self.traci.person.getSpeed(person) >= 0.1:
                continue
            # SUMO queues pedestrians on internal walking areas, not only on
            # the route's start edge. Use the actual upcoming crossing there.
            next_edge = self.traci.person.getNextEdge(person)
            crossing = next_edge if next_edge in counts else None
            if crossing is None:
                assigned = self.pedestrian_meta.get(person, (None, None))[1]
                if assigned in counts and road == CROSSING_APPROACHES[assigned]:
                    crossing = assigned
            if crossing is not None:
                counts[crossing] += 1
                oldest[crossing] = max(oldest[crossing], self.traci.person.getWaitingTime(person))
        self._last_crossing_max_waits = list(oldest.values())
        return {
            agent: (
                [min(1.0, counts[f":{agent}_c{i}"] / 10.0) for i in range(4)],
                [min(1.0, oldest[f":{agent}_c{i}"] / 120.0) for i in range(4)],
                [min(1.0, occupancies[f":{agent}_c{i}"] / 5.0) for i in range(4)],
            ) for agent in AGENTS
        }

    def _pedestrian_counts(self, agent: str) -> list[float]:
        return self._pedestrian_features()[agent][0]

    def observe(self) -> torch.Tensor:
        output = []
        local_queues: dict[str, float] = {}
        queue_values_by_agent: dict[str, list[float]] = {}
        density_values_by_agent: dict[str, list[float]] = {}
        pedestrian_features_by_agent = self._pedestrian_features()
        phase_by_agent: dict[str, int] = {}
        for agent in AGENTS:
            lanes = list(dict.fromkeys(self.traci.trafficlight.getControlledLanes(agent)))
            incoming = [lane for lane in lanes if not lane.startswith(":")]
            queue_values = [
                min(1.0, self.traci.lane.getLastStepHaltingNumber(lane) / 20.0)
                for lane in incoming[:4]
            ]
            while len(queue_values) < 4:
                queue_values.append(0.0)
            density_values = [
                min(1.0, self.traci.lane.getLastStepVehicleNumber(lane) / 25.0)
                for lane in incoming[:4]
            ]
            while len(density_values) < 4:
                density_values.append(0.0)
            local_queues[agent] = sum(queue_values) / 4
            queue_values_by_agent[agent] = queue_values
            density_values_by_agent[agent] = density_values
            phase_by_agent[agent] = self.traci.trafficlight.getPhase(agent)
        for agent in AGENTS:
            counts, oldest, occupancy = pedestrian_features_by_agent[agent]
            phase = phase_by_agent[agent]
            vehicle_features = queue_values_by_agent[agent] + density_values_by_agent[agent]
            pedestrian_features = counts + oldest + occupancy
            if self.observation_mode == "vehicle":
                pedestrian_features = [0.0] * 12
            elif self.observation_mode == "pedestrian":
                vehicle_features = [0.0] * 8
            phase_features = [1.0 if phase in (0, 3) else 0.0, phase / 5.0]
            neighbors = [
                local_queues[other]
                for other in AGENTS
                if other != agent
            ]
            values = (
                vehicle_features
                + pedestrian_features
                + phase_features
                + [min(1.0, self.phase_age[agent] / MAX_GREEN)]
                + neighbors
            )
            output.append(values)
        frame = torch.tensor(output, dtype=torch.float32)
        return self._stack_frame(frame)

    def _stack_frame(self, frame: torch.Tensor) -> torch.Tensor:
        count = getattr(self, "frame_stack", 1)
        if count == 1:
            return frame
        if not self._frame_history:
            self._frame_history = [frame.clone() for _ in range(count)]
        elif self._frame_time != self.time:
            self._frame_history = self._frame_history[1:] + [frame.clone()]
        self._frame_time = self.time
        return torch.cat(self._frame_history, dim=1)

    def pedestrian_actuated_actions(self) -> torch.Tensor:
        """The recorded baseline's request rule at legal five-second decisions.

        Preserve its assigned-crossing, stopped-person and road!=crossing
        semantics, including holding if the current phase is also requested.
        The baseline itself checks every second; this environment checks every5s.
        """
        active = set(self.traci.person.getIDList())
        masks = self.action_masks()
        actions = []
        for index, agent in enumerate(AGENTS):
            requests = {self.crossing_phases[agent][crossing]
                for person, (junction, crossing) in self.pedestrian_meta.items()
                if junction == agent and person in active
                and self.traci.person.getSpeed(person) < 0.1
                and self.traci.person.getRoadID(person) != crossing
                and crossing in self.crossing_phases[agent]}
            phase = self.traci.trafficlight.getPhase(agent)
            actions.append(int(bool(requests) and phase not in requests and bool(masks[index, 1])))
        return torch.tensor(actions, dtype=torch.long)

    def action_masks(self) -> torch.Tensor:
        masks = []
        for agent in AGENTS:
            phase = self.traci.trafficlight.getPhase(agent)
            can_switch = phase in (0, 3) and self.phase_age[agent] >= MIN_GREEN
            if self.action_size == 2:
                masks.append([True, can_switch])
            else:
                can_extend = phase in (0, 3) and self.phase_age[agent] < MAX_GREEN
                masks.append([True, can_switch, can_extend, can_extend])
        return torch.tensor(masks, dtype=torch.bool)

    def _reward(self, vehicle_wait: float, pedestrian_wait: float, switches: int) -> float:
        if self.reward_mode == "vehicle":
            value = self.vehicle_weight * vehicle_wait / VEHICLE_SCALE
        elif self.reward_mode == "raw":
            value = (
                self.vehicle_weight * vehicle_wait
                + self.pedestrian_weight * pedestrian_wait
            )
        elif self.reward_mode in ("vehicle_pedestrian", "normalized"):
            urgency = (
                URGENCY_COEFFICIENT * sum((wait / URGENCY_WAIT_SCALE) ** 2
                    for wait in getattr(self, "_last_crossing_max_waits", []))
                if self.reward_mode == "vehicle_pedestrian" else 0.0
            )
            value = (
                self.vehicle_weight * vehicle_wait / VEHICLE_SCALE
                + self.pedestrian_weight * (pedestrian_wait / PEDESTRIAN_SCALE + urgency)
            )
        elif self.reward_mode == "constrained":
            target = VEHICLE_SCALE * 1.05
            value = pedestrian_wait + 0.25 * vehicle_wait
            value += 10.0 * max(0.0, vehicle_wait - target)
        elif self.reward_mode == "queue":
            value = (
                self.vehicle_weight * vehicle_wait
                + self.pedestrian_weight * pedestrian_wait
            )
            value += 5.0 * sum(
                self.traci.lane.getLastStepHaltingNumber(lane)
                for agent in AGENTS
                for lane in set(self.traci.trafficlight.getControlledLanes(agent))
                if not lane.startswith(":")
            )
        else:
            value = (
                self.vehicle_weight * vehicle_wait
                + self.pedestrian_weight * pedestrian_wait
            )
        return -value - self.switch_penalty * switches

    def _spawn_pedestrians(self, now: int) -> None:
        if now >= 10 and now % self.pedestrian_interval == 0:
            for junction, start, crossing, end in PEDESTRIAN_ROUTES:
                person_id = f"train-ped-{self.pedestrian_id}"
                self.traci.person.add(person_id, start, 0, now)
                self.traci.person.appendWalkingStage(
                    person_id, [start, crossing, end], 0, self.pedestrian_speed
                )
                self.pedestrian_meta[person_id] = (junction, crossing)
                self.waiting_by_crossing.setdefault(crossing, 0.0)
                self.pedestrian_id += 1
                self.pedestrian_arrivals += 1

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        switches = 0
        for index, agent in enumerate(AGENTS):
            phase = self.traci.trafficlight.getPhase(agent)
            if phase != self.last_phase[agent]:
                self.phase_age[agent] = 0
                self.last_phase[agent] = phase
            self.phase_age[agent] += DECISION_INTERVAL
            if phase in (0, 3) and self.phase_age[agent] >= MAX_GREEN:
                self.traci.trafficlight.setPhaseDuration(agent, 0)
                self.phase_age[agent] = 0
            elif int(actions[index]) == 1 and self.action_masks()[index, 1]:
                self.traci.trafficlight.setPhaseDuration(agent, 0)
                self.phase_age[agent] = 0
                switches += 1
            elif (
                int(actions[index]) in (2, 3)
                and self.action_masks()[index, int(actions[index])]
            ):
                extension = 5 * (int(actions[index]) - 1)
                remaining = max(
                    0.0,
                    self.traci.trafficlight.getNextSwitch(agent)
                    - self.traci.simulation.getTime(),
                )
                duration = min(
                    remaining + extension,
                    max(0, MAX_GREEN - self.phase_age[agent]),
                )
                if duration > remaining:
                    self.traci.trafficlight.setPhaseDuration(agent, duration)
        step_vehicle_wait = 0.0
        step_pedestrian_wait = 0.0
        for _ in range(DECISION_INTERVAL):
            self._spawn_pedestrians(int(self.traci.simulation.getTime()))
            self.traci.simulationStep()
            vehicles = self.traci.vehicle.getIDList()
            persons = self.traci.person.getIDList()
            departed = self.traci.simulation.getDepartedIDList()
            arrived_vehicles = self.traci.simulation.getArrivedIDList()
            arrived_people = self.traci.simulation.getArrivedPersonIDList()
            self.vehicle_arrivals += len(departed)
            self.vehicle_completed += len(arrived_vehicles)
            self.pedestrian_completed += len(arrived_people)
            vehicle_wait = sum(
                self.traci.vehicle.getSpeed(vehicle) < 0.1 for vehicle in vehicles
            )
            pedestrian_wait = sum(
                self.traci.person.getSpeed(person) < 0.1 for person in persons
            )
            step_vehicle_wait += vehicle_wait
            step_pedestrian_wait += pedestrian_wait
            self.vehicle_waiting += vehicle_wait
            self.pedestrian_waiting += pedestrian_wait
            for vehicle in vehicles:
                wait = float(self.traci.vehicle.getSpeed(vehicle) < 0.1)
                self.vehicle_wait_by_id[vehicle] = (
                    self.vehicle_wait_by_id.get(vehicle, 0.0) + wait
                )
                time_loss = self.traci.vehicle.getTimeLoss(vehicle)
                previous_time_loss = self.vehicle_time_loss_by_id.get(vehicle, 0.0)
                self.vehicle_time_loss += max(0.0, time_loss - previous_time_loss)
                self.vehicle_time_loss_by_id[vehicle] = time_loss
            for person in persons:
                if self.traci.person.getSpeed(person) < 0.1:
                    self.pedestrian_wait_by_id[person] = (
                        self.pedestrian_wait_by_id.get(person, 0.0) + 1.0
                    )
                    crossing = self.pedestrian_meta[person][1]
                    self.waiting_by_crossing[crossing] = (
                        self.waiting_by_crossing.get(crossing, 0.0) + 1.0
                    )
            for vehicle in arrived_vehicles:
                self.vehicle_completed_waiting += self.vehicle_wait_by_id.pop(
                    vehicle, 0.0
                )
                self.vehicle_completed_time_loss += (
                    self.vehicle_time_loss_by_id.pop(vehicle, 0.0)
                )
            for person in arrived_people:
                waiting = self.pedestrian_wait_by_id.pop(person, 0.0)
                self.pedestrian_completed_waiting += waiting
                self.pedestrian_completed_waits.append(waiting)
                crossing = self.pedestrian_meta[person][1]
                self.pedestrian_waits_by_crossing.setdefault(
                    crossing, []
                ).append(waiting)
        self.time += DECISION_INTERVAL
        # observe() refreshes current per-crossing waits before reward shaping.
        observation = self.observe()
        return (
            observation,
            self._reward(step_vehicle_wait, step_pedestrian_wait, switches),
            self.time >= self.duration,
        )

    def close(self) -> None:
        self.traci.close()

    def metrics(self) -> dict[str, object]:
        active_vehicles = len(self.traci.vehicle.getIDList())
        active_person_ids = self.traci.person.getIDList()
        active_people = len(active_person_ids)
        pedestrian_waits = [
            self.pedestrian_wait_by_id.get(person_id, 0.0)
            for person_id in active_person_ids
        ]
        all_pedestrian_waits = pedestrian_waits + self.pedestrian_completed_waits
        active_waits_by_crossing: dict[str, list[float]] = {}
        for person_id in active_person_ids:
            waiting = self.pedestrian_wait_by_id.get(person_id, 0.0)
            crossing = self.pedestrian_meta[person_id][1]
            active_waits_by_crossing.setdefault(crossing, []).append(waiting)
        p95_by_crossing = {}
        for route in PEDESTRIAN_ROUTES:
            crossing = route[2]
            p95_by_crossing[crossing] = _p95(
                self.pedestrian_waits_by_crossing.get(crossing, [])
                + active_waits_by_crossing.get(crossing, [])
            )
        return {
            "vehicle_departed": self.vehicle_arrivals,
            "vehicle_completed": self.vehicle_completed,
            "vehicle_unfinished": active_vehicles,
            "pedestrian_departed": self.pedestrian_arrivals,
            "pedestrian_completed": self.pedestrian_completed,
            "pedestrian_unfinished": active_people,
            "vehicle_waiting_seconds": self.vehicle_waiting,
            "pedestrian_waiting_seconds": self.pedestrian_waiting,
            "vehicle_time_loss_seconds": self.vehicle_time_loss,
            "vehicle_completed_time_loss_seconds": self.vehicle_completed_time_loss,
            "vehicle_average_waiting_seconds": self.vehicle_waiting
            / max(self.vehicle_arrivals, 1),
            "vehicle_completed_average_waiting_seconds": self.vehicle_completed_waiting
            / max(self.vehicle_completed, 1),
            "vehicle_completed_average_time_loss_seconds": self.vehicle_completed_time_loss
            / max(self.vehicle_completed, 1),
            "pedestrian_average_waiting_seconds": self.pedestrian_waiting
            / max(self.pedestrian_arrivals, 1),
            "pedestrian_completed_average_waiting_seconds": self.pedestrian_completed_waiting
            / max(self.pedestrian_completed, 1),
            "pedestrian_p95_waiting_seconds": _p95(all_pedestrian_waits),
            "pedestrian_max_waiting_seconds": max(all_pedestrian_waits, default=0.0),
            "pedestrian_completed_p95_waiting_seconds": _p95(
                self.pedestrian_completed_waits
            ),
            "pedestrian_p95_by_crossing": p95_by_crossing,
            "waiting_by_crossing": dict(self.waiting_by_crossing),
        }


class Actor(nn.Module):
    def __init__(self, action_size: int = ACTION_SIZE, observation_size: int = OBSERVATION_SIZE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
        )

    def distribution(
        self, observation: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Categorical:
        logits = self.network(observation)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


class RND(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
        )
        for parameter in self.target.parameters():
            parameter.requires_grad = False

    def error(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self.target(observation)
        return (self.predictor(observation) - target).pow(2).mean(dim=-1)


def _gae(rewards, values, next_value, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    last = torch.zeros((), dtype=torch.float32)
    for index in reversed(range(len(rewards))):
        future = next_value if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + gamma * future - values[index]
        last = delta + gamma * gae_lambda * last
        advantages[index] = last
    return advantages


def _save_checkpoint(algorithm, actors, critics, path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": algorithm,
            "actors": [actor.state_dict() for actor in actors],
            "critics": [critic.state_dict() for critic in critics],
            "metadata": metadata,
        },
        path,
    )


def train(
    algorithm: str,
    episodes: int,
    seed: int,
    checkpoint: Path,
    horizon: int = TRAINING_HORIZON,
    snapshot_dir: Path | None = None,
    reward_mode: str = "raw",
    vehicle_weight: float = 1.0,
    pedestrian_weight: float = 1.0,
    switch_penalty: float = DEFAULT_SWITCH_PENALTY,
    observation_mode: str = "full",
    action_size: int = ACTION_SIZE,
    learning_rate: float = 3e-4,
    ppo_epochs: int = 5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coefficient: float = 0.01,
    rnd_coefficient: float = 0.0,
    green_duration: int = 37,
    pedestrian_speed: float = 1.4,
    pedestrian_interval: int = 20,
    vehicle_period: int = 6,
    snapshot_interval: int = 500,
    checkpoint_callback: Callable[[Path, int], None] | None = None,
    skip_build: bool = False,
    randomize_demand: bool = False,
    frame_stack: int = 1,
    bc_episodes: int = 0,
) -> None:
    if algorithm not in ("mappo", "ippo"):
        raise ValueError("algorithm must be mappo or ippo")
    if action_size not in ACTION_SIZES:
        raise ValueError(f"action_size must be one of {ACTION_SIZES}")
    if episodes <= 0 or horizon <= 0:
        raise ValueError("episodes and horizon must be positive")
    if snapshot_interval < 0:
        raise ValueError("snapshot_interval must be non-negative")
    if frame_stack not in (1, 3) or bc_episodes < 0:
        raise ValueError("frame_stack must be 1 or 3 and bc_episodes non-negative")
    if bc_episodes and action_size != 2:
        raise ValueError("BC requires the two-action space")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if not skip_build:
        build(green_duration, pedestrian_speed, vehicle_period)
    observation_size = OBSERVATION_SIZE * frame_stack
    actors = [Actor(action_size, observation_size) for _ in AGENTS]
    critics = (
        [Critic(observation_size * 3)]
        if algorithm == "mappo"
        else [Critic(observation_size) for _ in AGENTS]
    )
    rnd = RND(observation_size * 3) if rnd_coefficient > 0 else None
    parameters = [p for model in actors + critics for p in model.parameters()]
    if rnd is not None:
        parameters.extend(rnd.predictor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    demand_random = random.Random(seed)
    temporary_networks = TemporaryDirectory(prefix="demand-", dir=checkpoint.parent) if randomize_demand else None
    network_cache: dict[int, Path] = {}
    schedule_path = checkpoint.parent / "demand-schedule.jsonl"
    if randomize_demand:
        schedule_path.write_text("", encoding="utf-8")
    metadata = {
        "observation_mode": observation_mode,
        "reward_mode": reward_mode,
        "vehicle_weight": vehicle_weight,
        "pedestrian_weight": pedestrian_weight,
        "switch_penalty": switch_penalty,
        "rnd_coefficient": rnd_coefficient,
        "action_size": action_size,
        "green_duration": green_duration,
        "pedestrian_speed": pedestrian_speed,
        "pedestrian_interval": pedestrian_interval,
        "vehicle_period": vehicle_period,
        "randomize_demand": randomize_demand,
        "vehicle_period_choices": list(VEHICLE_PERIOD_CHOICES) if randomize_demand else [vehicle_period],
        "pedestrian_interval_choices": list(PEDESTRIAN_INTERVAL_CHOICES) if randomize_demand else [pedestrian_interval],
        "reward_objective_version": "v5-bc-frame-stack" if bc_episodes or frame_stack == 3 else "v4-upstream-calibrated-switching",
        "observation_size": observation_size,
        "base_observation_size": OBSERVATION_SIZE,
        "frame_stack": frame_stack,
        "frame_order": "oldest-to-newest; reset padded with three initial-frame copies",
        "centralized_critic_input_size": observation_size * 3,
        "bc_episodes": bc_episodes,
        "bc_sampling": "balanced-legal-action-minibatches" if bc_episodes else None,
        "critic_warmup_episodes": 5 if bc_episodes else 0,
        "learning_rate": learning_rate,
        "observation_layout": ["vehicle_queue_4", "approach_vehicle_density_4", "pedestrian_waiting_count_4", "oldest_current_wait_4", "crossing_occupancy_4", "phase_2", "phase_age_1", "neighbor_queue_2"],
        "approach_vehicle_density_scale": 25.0,
        "pedestrian_detection": "next-crossing-walking-area-with-assigned-start-fallback",
        "urgency_coefficient": URGENCY_COEFFICIENT,
        "urgency_wait_scale": URGENCY_WAIT_SCALE,
        "urgency_aggregation": "sum over 12 crossings once per five-second reward step",
        "vehicle_scale": VEHICLE_SCALE,
        "pedestrian_scale": PEDESTRIAN_SCALE,
    }

    if bc_episodes:
        if __package__:
            from .marl_bc import warm_start
        else:
            from marl_bc import warm_start
        def bc_environment(simulation_seed):
            return SignalEnvironment(simulation_seed, horizon, reward_mode,
                vehicle_weight, pedestrian_weight, switch_penalty, observation_mode,
                action_size, green_duration, pedestrian_speed, pedestrian_interval,
                frame_stack=frame_stack)
        bc_report = warm_start(actors, critics, bc_environment, bc_episodes, seed, gamma, algorithm)
        (checkpoint.parent / "bc-metrics.json").write_text(json.dumps(bc_report, indent=2), encoding="utf-8")
        bc_path = (snapshot_dir or checkpoint.parent) / "episode-0-bc.pt"
        _save_checkpoint(algorithm, actors, critics, bc_path,
                         {**metadata, "training_episode": 0, "stage": "bc-and-critic-warmup"})
        if checkpoint_callback is not None:
            checkpoint_callback(bc_path, 0)

    for episode in range(episodes):
        episode_period = demand_random.choice(VEHICLE_PERIOD_CHOICES) if randomize_demand else vehicle_period
        episode_interval = demand_random.choice(PEDESTRIAN_INTERVAL_CHOICES) if randomize_demand else pedestrian_interval
        configuration = None
        if temporary_networks is not None:
            if episode_period not in network_cache:
                target = Path(temporary_networks.name) / f"vehicle-period-{episode_period}"
                build(green_duration, pedestrian_speed, episode_period, output_dir=target)
                network_cache[episode_period] = target / "simulation.sumocfg"
            configuration = network_cache[episode_period]
            with schedule_path.open("a", encoding="utf-8") as schedule:
                schedule.write(json.dumps({"episode": episode+1, "simulation_seed": seed+episode,
                    "vehicle_period": episode_period, "pedestrian_interval": episode_interval})+"\n")
        environment = SignalEnvironment(
            seed + episode,
            horizon,
            reward_mode,
            vehicle_weight,
            pedestrian_weight,
            switch_penalty,
            observation_mode,
            action_size,
            green_duration,
            pedestrian_speed,
            episode_interval,
            configuration,
            frame_stack=frame_stack,
        )
        observation = environment.reset()
        rollout = []
        try:
            done = False
            while not done:
                masks = environment.action_masks()
                distributions = [
                    actor.distribution(observation[index], masks[index])
                    for index, actor in enumerate(actors)
                ]
                actions = torch.stack(
                    [distribution.sample() for distribution in distributions]
                )
                log_probs = torch.stack(
                    [
                        distribution.log_prob(actions[index])
                        for index, distribution in enumerate(distributions)
                    ]
                )
                state = observation.flatten()
                values = torch.stack(
                    [
                        critic(state if algorithm == "mappo" else observation[index])
                        for index, critic in enumerate(critics)
                    ]
                )
                next_observation, reward, done = environment.step(actions)
                intrinsic = 0.0
                if rnd is not None:
                    intrinsic = rnd.error(observation.flatten()).detach().item()
                rollout.append(
                    (
                        observation,
                        masks,
                        actions,
                        log_probs,
                        values,
                        reward + rnd_coefficient * intrinsic,
                        done,
                    )
                )
                observation = next_observation
        finally:
            environment.close()

        states = torch.stack([item[0] for item in rollout])
        actions = torch.stack([item[2] for item in rollout])
        old_log_probs = torch.stack([item[3] for item in rollout])
        rewards = torch.tensor([item[5] for item in rollout], dtype=torch.float32)
        values = torch.stack([item[4] for item in rollout])
        advantages = []
        returns = []
        for index in range(len(critics)):
            value = values[:, index]
            next_value = torch.zeros(())
            advantage = _gae(rewards, value, next_value, gamma, gae_lambda)
            advantages.append((advantage - advantage.mean()) / (advantage.std() + 1e-8))
            returns.append(advantage + value)

        for _ in range(ppo_epochs):
            policy_loss = torch.zeros(())
            value_loss = torch.zeros(())
            entropy_loss = torch.zeros(())
            for index, actor in enumerate(actors):
                distributions = actor.distribution(
                    states[:, index, :],
                    torch.stack([item[1][index] for item in rollout]),
                )
                new_log_probs = distributions.log_prob(actions[:, index])
                ratio = torch.exp(new_log_probs - old_log_probs[:, index].detach())
                clipped = torch.clamp(
                    ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
                )
                policy_loss -= torch.minimum(
                    ratio * advantages[0 if algorithm == "mappo" else index].detach(),
                    clipped * advantages[0 if algorithm == "mappo" else index].detach(),
                ).mean()
                entropy_loss += distributions.entropy().mean()
            for index, critic in enumerate(critics):
                critic_input = (
                    states.flatten(start_dim=1)
                    if algorithm == "mappo"
                    else states[:, index, :]
                )
                value_loss += nn.functional.mse_loss(
                    critic(critic_input), returns[index].detach()
                )
            loss = policy_loss + 0.5 * value_loss - entropy_coefficient * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
        if rnd is not None:
            rnd_loss = rnd.error(states.flatten(start_dim=1)).mean()
            optimizer.zero_grad()
            rnd_loss.backward()
            optimizer.step()

        if (
            snapshot_dir is not None
            and snapshot_interval
            and (episode + 1) % snapshot_interval == 0
        ):
            _save_checkpoint(
                algorithm,
                actors,
                critics,
                snapshot_dir / f"episode-{episode + 1}.pt",
                {**metadata, "training_episode": episode + 1, "stage": "ppo"},
            )
            if checkpoint_callback is not None:
                checkpoint_callback(snapshot_dir / f"episode-{episode + 1}.pt", episode + 1)
        print(f"episode={episode + 1}/{episodes} reward={rewards.sum().item():.1f}")
    _save_checkpoint(algorithm, actors, critics, checkpoint, {**metadata, "training_episode": episodes, "stage": "ppo"})
    if temporary_networks is not None:
        temporary_networks.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("mappo", "ippo"), default="mappo")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=TRAINING_HORIZON)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=GENERATED / "mappo.pt")
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument(
        "--reward-mode",
        choices=("vehicle", "vehicle_pedestrian", "raw", "normalized", "constrained", "queue"),
    )
    parser.add_argument("--action-size", choices=(2, 4), type=int, default=ACTION_SIZE)
    parser.add_argument("--vehicle-weight", type=float, default=1.0)
    parser.add_argument("--pedestrian-weight", type=float, default=1.0)
    parser.add_argument("--switch-penalty", type=float, default=DEFAULT_SWITCH_PENALTY)
    parser.add_argument("--observation-mode", choices=("full", "vehicle", "pedestrian"))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--rnd-coefficient", type=float, default=0.0)
    parser.add_argument("--green-duration", type=int, default=37)
    parser.add_argument("--pedestrian-speed", type=float, default=1.4)
    parser.add_argument("--snapshot-interval", type=int, default=500)
    parser.add_argument("--pedestrian-interval", type=int, default=20)
    parser.add_argument("--vehicle-period", type=int, default=6)
    parser.add_argument("--randomize-demand", action="store_true")
    parser.add_argument("--frame-stack", type=int, choices=(1, 3), default=1)
    parser.add_argument("--bc-episodes", type=int, default=0)
    args = parser.parse_args()
    train(
        args.algorithm,
        args.episodes,
        args.seed,
        args.checkpoint,
        args.horizon,
        args.snapshot_dir,
        args.reward_mode or ("normalized" if args.algorithm == "mappo" else "raw"),
        args.vehicle_weight,
        args.pedestrian_weight,
        args.switch_penalty,
        args.observation_mode or "full",
        args.action_size,
        args.learning_rate,
        args.ppo_epochs,
        args.gamma,
        args.gae_lambda,
        args.clip_epsilon,
        args.entropy_coefficient,
        args.rnd_coefficient,
        args.green_duration,
        args.pedestrian_speed,
        pedestrian_interval=args.pedestrian_interval,
        vehicle_period=args.vehicle_period,
        snapshot_interval=args.snapshot_interval,
        randomize_demand=args.randomize_demand,
        frame_stack=args.frame_stack,
        bc_episodes=args.bc_episodes,
    )

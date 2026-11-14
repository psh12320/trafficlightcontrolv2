"""Train canonical clipped-PPO IPPO or MAPPO signal controllers."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.distributions import Categorical

if __package__:
    from .build_network import GENERATED, build
    from .run_baseline import PEDESTRIAN_ROUTES, _crossing_phase_map
    from .run_simulation import _import_traci, _sumo_binary
else:
    from build_network import GENERATED, build
    from run_baseline import PEDESTRIAN_ROUTES, _crossing_phase_map
    from run_simulation import _import_traci, _sumo_binary


AGENTS = ("J0", "J1", "J2")
OBSERVATION_SIZE = 13
ACTION_SIZE = 4
ACTION_SIZES = (2, 4)
DECISION_INTERVAL = 5
TRAINING_HORIZON = 120
MIN_GREEN = 10
MAX_GREEN = 45
VEHICLE_SCALE = 20.8
PEDESTRIAN_SCALE = 81.35


class SignalEnvironment:
    """TraCI environment with safe signal timing actions."""

    def __init__(
        self,
        seed: int,
        duration: int = TRAINING_HORIZON,
        reward_mode: str = "raw",
        vehicle_weight: float = 1.0,
        pedestrian_weight: float = 1.0,
        switch_penalty: float = 0.1,
        observation_mode: str = "full",
        action_size: int = ACTION_SIZE,
        green_duration: int = 37,
        pedestrian_speed: float = 1.4,
    ):
        self.traci = _import_traci()
        self.seed = seed
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
        self.waiting_by_crossing: dict[str, float] = {}
        self.phase_age = {agent: 0 for agent in AGENTS}
        self.last_phase = {agent: None for agent in AGENTS}
        self.crossing_phases: dict[str, dict[str, int]] = {}

    def reset(self) -> torch.Tensor:
        self.traci.start(
            [
                _sumo_binary("sumo"),
                "-c",
                str(GENERATED / "simulation.sumocfg"),
                "--seed",
                str(self.seed),
                "--end",
                str(self.duration),
                "--step-length",
                "1",
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
        self.waiting_by_crossing = {}
        self.phase_age = {agent: 0 for agent in AGENTS}
        self.last_phase = {
            agent: self.traci.trafficlight.getPhase(agent) for agent in AGENTS
        }
        self.crossing_phases = {
            agent: _crossing_phase_map(self.traci, agent) for agent in AGENTS
        }
        return self.observe()

    def _pedestrian_counts(self, agent: str) -> list[float]:
        counts = {f":{agent}_c{index}": 0 for index in range(4)}
        for person in self.traci.person.getIDList():
            road = self.traci.person.getRoadID(person)
            for crossing in counts:
                if road == crossing and self.traci.person.getSpeed(person) < 0.1:
                    counts[crossing] += 1
        return [min(1.0, counts[key] / 10.0) for key in counts]

    def observe(self) -> torch.Tensor:
        output = []
        local_queues: dict[str, float] = {}
        queue_values_by_agent: dict[str, list[float]] = {}
        pedestrian_values_by_agent: dict[str, list[float]] = {}
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
            local_queues[agent] = sum(queue_values) / 4
            queue_values_by_agent[agent] = queue_values
            pedestrian_values_by_agent[agent] = self._pedestrian_counts(agent)
            phase_by_agent[agent] = self.traci.trafficlight.getPhase(agent)
        for agent in AGENTS:
            pedestrian_values = pedestrian_values_by_agent[agent]
            phase = phase_by_agent[agent]
            vehicle_features = queue_values_by_agent[agent]
            pedestrian_features = pedestrian_values
            if self.observation_mode == "vehicle":
                pedestrian_features = [0.0] * 4
            elif self.observation_mode == "pedestrian":
                vehicle_features = [0.0] * 4
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
        return torch.tensor(output, dtype=torch.float32)

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
            value = self.vehicle_weight * vehicle_wait
        elif self.reward_mode in ("vehicle_pedestrian", "raw"):
            value = (
                self.vehicle_weight * vehicle_wait
                + self.pedestrian_weight * pedestrian_wait
            )
        elif self.reward_mode == "normalized":
            value = (
                self.vehicle_weight * vehicle_wait / VEHICLE_SCALE
                + self.pedestrian_weight * pedestrian_wait / PEDESTRIAN_SCALE
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

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        if self.time >= 10 and self.time % 20 == 0:
            for junction, start, crossing, end in PEDESTRIAN_ROUTES:
                person_id = f"train-ped-{self.pedestrian_id}"
                self.traci.person.add(person_id, start, 0, self.time)
                self.traci.person.appendWalkingStage(
                    person_id, [start, crossing, end], 0, self.pedestrian_speed
                )
                self.pedestrian_meta[person_id] = (junction, crossing)
                self.waiting_by_crossing.setdefault(crossing, 0.0)
                self.pedestrian_id += 1
                self.pedestrian_arrivals += 1
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
        self.time += DECISION_INTERVAL
        return (
            self.observe(),
            self._reward(step_vehicle_wait, step_pedestrian_wait, switches),
            self.time >= self.duration,
        )

    def close(self) -> None:
        self.traci.close()

    def metrics(self) -> dict[str, object]:
        active_vehicles = len(self.traci.vehicle.getIDList())
        active_people = len(self.traci.person.getIDList())
        pedestrian_waits = list(self.pedestrian_wait_by_id.values())
        all_pedestrian_waits = pedestrian_waits + self.pedestrian_completed_waits
        p95 = (
            statistics.quantiles(all_pedestrian_waits, n=20, method="inclusive")[18]
            if len(all_pedestrian_waits) > 1
            else (all_pedestrian_waits[0] if all_pedestrian_waits else 0.0)
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
            "pedestrian_p95_waiting_seconds": p95,
            "pedestrian_completed_p95_waiting_seconds": (
                statistics.quantiles(
                    self.pedestrian_completed_waits,
                    n=20,
                    method="inclusive",
                )[18]
                if len(self.pedestrian_completed_waits) > 1
                else (
                    self.pedestrian_completed_waits[0]
                    if self.pedestrian_completed_waits
                    else 0.0
                )
            ),
            "waiting_by_crossing": dict(self.waiting_by_crossing),
        }


class Actor(nn.Module):
    def __init__(self, action_size: int = ACTION_SIZE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 64),
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
    switch_penalty: float = 0.1,
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
    snapshot_interval: int = 500,
    checkpoint_callback: Callable[[Path, int], None] | None = None,
) -> None:
    if algorithm not in ("mappo", "ippo"):
        raise ValueError("algorithm must be mappo or ippo")
    if action_size not in ACTION_SIZES:
        raise ValueError(f"action_size must be one of {ACTION_SIZES}")
    if episodes <= 0 or horizon <= 0:
        raise ValueError("episodes and horizon must be positive")
    if snapshot_interval < 0:
        raise ValueError("snapshot_interval must be non-negative")
    torch.manual_seed(seed)
    build(green_duration, pedestrian_speed)
    actors = [Actor(action_size) for _ in AGENTS]
    critics = (
        [Critic(OBSERVATION_SIZE * 3)]
        if algorithm == "mappo"
        else [Critic(OBSERVATION_SIZE) for _ in AGENTS]
    )
    rnd = RND(OBSERVATION_SIZE * 3) if rnd_coefficient > 0 else None
    parameters = [p for model in actors + critics for p in model.parameters()]
    if rnd is not None:
        parameters.extend(rnd.predictor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
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
    }

    for episode in range(episodes):
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
                metadata,
            )
            if checkpoint_callback is not None:
                checkpoint_callback(snapshot_dir / f"episode-{episode + 1}.pt", episode + 1)
        print(f"episode={episode + 1}/{episodes} reward={rewards.sum().item():.1f}")
    _save_checkpoint(algorithm, actors, critics, checkpoint, metadata)


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
    parser.add_argument("--switch-penalty", type=float, default=0.1)
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
        args.snapshot_interval,
    )

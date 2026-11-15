from types import SimpleNamespace

import pytest
import torch

from scripts import train_marl as marl
from scripts.marl_bc import discounted_returns, fit_actors, warm_start


def test_frame_stack_initial_padding_chronology_and_same_tick_reads():
    e = marl.SignalEnvironment.__new__(marl.SignalEnvironment)
    e.frame_stack, e.time, e._frame_time, e._frame_history = 3, 0, None, []
    first = torch.arange(75, dtype=torch.float32).reshape(3, 25)
    assert torch.equal(e._stack_frame(first), torch.cat([first]*3, dim=1))
    # Re-reading observations cannot manufacture additional temporal frames.
    assert torch.equal(e._stack_frame(first+9), torch.cat([first]*3, dim=1))
    e.time = 5
    assert torch.equal(e._stack_frame(first+1), torch.cat([first, first, first+1], dim=1))
    e.time = 10
    assert torch.equal(e._stack_frame(first+2), torch.cat([first, first+1, first+2], dim=1))
    e.time, e._frame_time, e._frame_history = 0, None, []
    assert torch.equal(e._stack_frame(first+7), torch.cat([first+7]*3, dim=1))
    assert marl.Actor(observation_size=75).network[0].in_features == 75
    assert marl.Critic(225).network[0].in_features == 225


def test_expert_preserves_baseline_competing_requests_and_legal_masks():
    e = marl.SignalEnvironment.__new__(marl.SignalEnvironment)
    persons = {'red': ('J0', ':J0_c1', 'N0_J0', 0.0),
               'both_green': ('J1', ':J1_c0', 'N1_J1', 0.0),
               'both_red': ('J1', ':J1_c1', 'N1_J1', 0.0),
               'too_early': ('J2', ':J2_c1', 'N2_J2', 0.0),
               'crossing': ('J0', ':J0_c0', ':J0_c0', 0.0),
               'moving': ('J0', ':J0_c0', 'N0_J0', 1.0)}
    e.pedestrian_meta = {p: v[:2] for p, v in persons.items()}
    e.traci = SimpleNamespace(person=SimpleNamespace(getIDList=lambda: list(persons),
        getRoadID=lambda p: persons[p][2], getSpeed=lambda p: persons[p][3]),
        trafficlight=SimpleNamespace(getPhase=lambda j: 0))
    e.crossing_phases = {j: {f':{j}_c0': 0, f':{j}_c1': 3} for j in marl.AGENTS}
    e.phase_age, e.action_size = {'J0': 10, 'J1': 10, 'J2': 5}, 2
    assert e.pedestrian_actuated_actions().tolist() == [1, 0, 0]


def test_discounted_returns_stop_at_episode_boundary():
    assert discounted_returns([1, 2, 3], 0.5).tolist() == pytest.approx([2.75, 3.5, 3])


def test_masked_cloning_learns_and_never_emits_illegal_switches():
    torch.set_num_threads(1)
    torch.manual_seed(731)
    observations = torch.randn(512, 3, 75)
    masks = torch.ones(512, 3, 2, dtype=torch.bool)
    masks[:80, :, 1] = False
    actions = (observations[:, :, 0] > 0).long()
    actions[:80] = 0
    actors = [marl.Actor(observation_size=75) for _ in marl.AGENTS]
    reports = fit_actors(actors, observations, masks, actions)
    for j, (actor, report) in enumerate(zip(actors, reports)):
        assert report['cross_entropy_by_epoch'][-1] < report['cross_entropy_by_epoch'][0]
        assert actor.distribution(observations[:80, j], masks[:80, j]).probs.argmax(-1).eq(0).all()


def test_critic_warmup_keeps_cloned_actors_and_improves_value_fit():
    torch.manual_seed(700)
    class Environment:
        def reset(self):
            self.t = 0
            return torch.zeros(3, 75)
        def action_masks(self): return torch.ones(3, 2, dtype=torch.bool)
        def pedestrian_actuated_actions(self): return torch.tensor([1, 0, 1])
        def step(self, actions):
            self.t += 1
            return torch.full((3, 75), self.t/10), -2.0, self.t == 2
        def close(self): pass
    actors = [marl.Actor(observation_size=75) for _ in marl.AGENTS]
    result = warm_start(actors, [marl.Critic(225)], lambda seed: Environment(), 2, 700, 0.99, 'mappo')
    assert result['bc_transitions_per_agent'] == 4
    assert result['critic_warmup_episodes'] == 5
    assert result['critic_warmup_actor_parameters_unchanged'] is True
    assert result['critic_fits'][0]['mse_after_warmup'] < result['critic_fits'][0]['mse_before_warmup']


def test_bc_learns_rare_switches_instead_of_majority_hold_collapse():
    torch.set_num_threads(1)
    torch.manual_seed(818)
    observations = torch.zeros(1024, 3, 75)
    observations[-4:, :, 0] = 1
    actions = torch.zeros(1024, 3, dtype=torch.long)
    actions[-4:] = 1
    masks = torch.ones(1024, 3, 2, dtype=torch.bool)
    masks[:200, :, 1] = False
    reports = fit_actors([marl.Actor(observation_size=75) for _ in marl.AGENTS], observations, masks, actions)
    assert all(r['switch_recall'] == 1 for r in reports)
    assert all(r['all_decision_accuracy'] > 0.99 for r in reports)

"""Real SUMO BC, frame-stack, critic-warmup and PPO smoke test."""
import argparse
from argparse import Namespace
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_simulation import _import_traci
_import_traci()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from scripts.run_marl_sweep import main as sweep
    sweep(Namespace(output_root=args.output, seed=9970, episodes=2, horizon=120,
        randomize_demand=False, switch_penalty=3.0, skip_build=False,
        pedestrian_speed=1.4, vehicle_period=6, pedestrian_interval=20,
        algorithms=['mappo'], reward_modes=['vehicle', 'vehicle_pedestrian'],
        action_sizes=[2], evaluation_seeds=[9901, 9902, 9903],
        evaluation_duration=120, snapshot_interval=1, learning_rate=1e-4,
        frame_stack=3, bc_episodes=2))
    import torch
    paths = list(args.output.glob('*/final.pt'))
    assert len(paths) == 2
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['metadata']['observation_size'] == 75
        assert saved['metadata']['frame_stack'] == 3
        assert saved['metadata']['bc_episodes'] == 2
        assert saved['metadata']['learning_rate'] == 1e-4
        assert saved['critics'][0]['network.0.weight'].shape[1] == 225
        bc = torch.load(path.parent/'snapshots/episode-0-bc.pt', map_location='cpu', weights_only=True)
        assert not all(torch.equal(v, saved['actors'][0][k]) for k, v in bc['actors'][0].items())
        report = json.loads((path.parent/'bc-metrics.json').read_text())
        assert report['bc_transitions_per_agent'] == 48
        assert report['critic_warmup_actor_parameters_unchanged'] is True
        with (path.parent/'checkpoint-metrics.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert sorted(int(r['training_episode']) for r in rows) == [0]*3+[1]*3+[2]*3
    print('V5_BC_FRAME_STACK_CRITIC_PPO_SMOKE_PASSED', flush=True)


if __name__ == '__main__': main()

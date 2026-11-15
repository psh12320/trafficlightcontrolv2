"""Small fixed-demand MAPPO smoke test, using diagnostic seeds only."""
import argparse
from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_simulation import _import_traci
_import_traci()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from scripts.run_marl_sweep import main as sweep
    sweep(Namespace(output_root=args.output,seed=9980,episodes=2,horizon=120,
        randomize_demand=False,switch_penalty=3.0,skip_build=False,
        pedestrian_speed=1.4,vehicle_period=6,pedestrian_interval=20,
        algorithms=['mappo'],reward_modes=['vehicle','vehicle_pedestrian'],
        action_sizes=[2],evaluation_seeds=[9901,9902,9903],
        evaluation_duration=120,snapshot_interval=1))
    import torch
    paths=list(args.output.glob('*/final.pt'));assert len(paths)==2
    for p in paths:
        saved=torch.load(p,map_location='cpu',weights_only=True)
        assert saved['metadata']['observation_size']==25
        assert saved['metadata']['switch_penalty']==3
        assert saved['metadata']['randomize_demand'] is False
        assert saved['critics'][0]['network.0.weight'].shape[1]==75
    print('V4_TWO_CONTROLLER_FIXED_DEMAND_SMOKE_PASSED',flush=True)


if __name__=='__main__':main()

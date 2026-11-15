from types import SimpleNamespace

import pytest

from scripts import train_marl as marl


def test_moving_platoons_are_visible_before_a_stopped_queue_forms():
    e = marl.SignalEnvironment.__new__(marl.SignalEnvironment)
    e.traci = SimpleNamespace(
        trafficlight=SimpleNamespace(getControlledLanes=lambda j: [j+str(i) for i in range(4)],
                                     getPhase=lambda j: 0),
        lane=SimpleNamespace(getLastStepHaltingNumber=lambda lane: 0,
            getLastStepVehicleNumber=lambda lane: [5,10,25,50][int(lane[-1])]))
    e.phase_age = {j: 10 for j in marl.AGENTS}
    e._pedestrian_features = lambda: {j: ([0.0]*4,[0.0]*4,[0.0]*4) for j in marl.AGENTS}
    e.observation_mode = "full"
    obs = e.observe()
    assert obs.shape == (3,25)
    assert obs[0,:4].tolist() == [0,0,0,0]
    assert obs[0,4:8].tolist() == pytest.approx([0.2,0.4,1,1])
    assert marl.DEFAULT_SWITCH_PENALTY == 3

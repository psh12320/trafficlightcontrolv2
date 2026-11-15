import csv
import json
import math

import pytest

from scripts.summarize_proposal_pilot import summarize


@pytest.mark.parametrize('action_size', [2, 4])
def test_confidence_interval_uses_independent_training_seeds(tmp_path, action_size):
    rows = []
    for training_seed, center in ((100, 10), (200, 20), (300, 30)):
        for test_seed, offset in ((11, -1), (12, 0), (13, 1)):
            rows.append({
                "scenario": "primary", "scenario_controller": f"ippo-vehicle-actions{action_size}-seed{training_seed}",
                "seed": test_seed, "duration_seconds": 3600,
                "vehicle_time_loss_seconds": (center+offset)*100,
                "vehicle_departed": 100, "vehicle_completed": 90,
                "vehicle_unfinished": 10, "pedestrian_unfinished": 5,
                "pedestrian_average_waiting_seconds": center+offset,
                "pedestrian_p95_by_crossing": json.dumps({f"crossing{i}": center+offset for i in range(12)}),
            })
            if action_size == 2:
                rows[-1]['pedestrian_max_waiting_seconds'] = center + offset + 200
    path = tmp_path/"results.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = summarize([path])
    metric = next(row for row in result if row["metric"] == "vehicle_mean_time_loss_seconds")
    assert metric["mean"] == 20
    assert metric["independent_n"] == 3
    assert metric["evaluation_n"] == 9
    assert metric["uncertainty_unit"] == "training_seed"
    assert metric["ci95_half_width"] == pytest.approx(4.302652729911275*10/math.sqrt(3))
    maxima = [r for r in result if r['metric']=='pedestrian_max_waiting_seconds']
    if action_size == 2:
        assert maxima[0]['mean'] == 220
        assert maxima[0]['maximum_observed'] == 231
    else:
        assert not maxima

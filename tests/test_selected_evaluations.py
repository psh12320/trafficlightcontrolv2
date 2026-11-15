import csv
from contextlib import contextmanager
from pathlib import Path

from scripts import run_selected_evaluations as selected


def test_scenario_preserves_different_baseline_and_learned_columns(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(selected, "build", lambda *args: builds.append(args))

    @contextmanager
    def pool(**kwargs):
        class InlinePool:
            def map(self, function, tasks):
                for controller, checkpoint, *_ in tasks:
                    row = {"seed": 11, "scenario_controller": controller,
                           "pedestrian_p95_by_crossing": {":J0_c0": 12.0}}
                    if checkpoint is None:
                        row["warmup_seconds"] = 60
                    else:
                        row["checkpoint"] = str(checkpoint)
                        row["vehicle_departed"] = 10
                    yield [row]
        yield InlinePool()

    monkeypatch.setattr(selected, "ProcessPoolExecutor", pool)
    output = selected.evaluate_scenario(
        "slow", [("ippo-vehicle-seed100", Path("selected.pt"))],
        [11], 3600, 0.9, 6, 20, tmp_path,
    )
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3
    assert rows[0]["warmup_seconds"] == "60"
    assert rows[2]["checkpoint"] == "selected.pt"
    assert rows[2]["vehicle_departed"] == "10"
    assert rows[0]["pedestrian_p95_by_crossing"] == '{":J0_c0": 12.0}'
    assert builds == [(37, 0.9, 6)]


def test_reused_baselines_are_preserved_without_new_baseline_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(selected, 'build', lambda *args: None)
    tasks_seen=[]
    @contextmanager
    def pool(**kwargs):
        class InlinePool:
            def map(self, function, tasks):
                for task in tasks:
                    tasks_seen.append(task)
                    assert task[1] is not None
                    yield [{'seed':41,'scenario_controller':task[0],
                            'pedestrian_p95_by_crossing':{':J0_c0':5.0}}]
        yield InlinePool()
    monkeypatch.setattr(selected,'ProcessPoolExecutor',pool)
    baselines=[{'scenario':'primary','scenario_controller':c,'seed':'41',
                'pedestrian_p95_by_crossing':'{":J0_c0": 12.0}',
                'vehicle_time_loss_seconds':'123.456'}
               for c in ['fixed_time','pedestrian_actuated']]
    output=selected.evaluate_scenario('primary',[('mappo-vehicle-actions2-seed100',Path('model.pt'))],
        [41],3600,1.4,6,20,tmp_path,baseline_rows=baselines)
    with output.open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==3 and len(tasks_seen)==1
    assert [r['vehicle_time_loss_seconds'] for r in rows[:2]]==['123.456','123.456']
    assert baselines[0]['seed']=='41'

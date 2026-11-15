"""Package result files and selected checkpoints for an authenticated download."""
import argparse
import hashlib
import json
import tarfile
import csv
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
status_path = args.root/"pilot-status.json"
complete = status_path.exists() and json.loads(status_path.read_text())["stage"] == "complete"
correction = args.root/"demand-correction-pending.json"
if complete and correction.exists():
    corrected = args.root/"demand-correction/high-corrected-results.csv"
    until = time.monotonic()+15
    correction_source = args.root/"demand-correction/source.json"
    while not correction_source.exists() and time.monotonic() < until:
        time.sleep(0.5)
    if not corrected.exists() or not correction_source.exists():
        raise RuntimeError("corrected high-demand evaluation has not finished")
    rows = []
    high_rows = []
    for path in (args.root/"low-results.csv", corrected):
        with path.open(newline="", encoding="utf-8") as file:
            part = list(csv.DictReader(file))
        if len(part) != 42:
            raise ValueError(f"expected 42 evaluations: {path}")
        if path == corrected:
            for row in part:
                row["scenario"] = "high"
                departed = row.get("pedestrian_departed") or row.get("pedestrian_arrivals")
                if int(departed) != 3588:
                    raise ValueError(f"high demand must inject 3588 pedestrians, got {departed}")
            high_rows = part
        rows.extend(part)
    diagnostics = args.root/"diagnostics"
    diagnostics.mkdir(exist_ok=True)
    original = args.root/"high-results.csv"
    if original.exists() and not (diagnostics/"invalid-high-results.csv").exists():
        original.replace(diagnostics/"invalid-high-results.csv")
    with original.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in high_rows for key in row}))
        writer.writeheader()
        writer.writerows(high_rows)
    with (args.root/"final-demand-sweep.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    (args.root/"demand-correction-applied.json").write_text(json.dumps({
        "reason": "12-second pedestrian arrivals must occur between 5-second policy decisions",
        "high_pedestrians_per_evaluation": 3588,
        "source": str(corrected), "applied_unix": time.time(),
        "invalidated_raw_results": "diagnostics/invalid-high-results.csv and high/*.json (initial run)",
        "accepted_raw_results": "demand-correction/high-corrected/*.json",
    }, indent=2))
selection_path = args.root/"selected-checkpoints.json"
selected = {Path(row["checkpoint"]).resolve() for row in json.loads(selection_path.read_text())} if selection_path.exists() else set()
if complete and correction.exists():
    manifest = {str(path.relative_to(args.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in args.root.rglob("*") if path.is_file()
                and path.name not in ("artifact-manifest.json", "pilot-status.json", "pilot.log")
                and (path.suffix in (".csv", ".json", ".log") or path.resolve() in selected)}
    (args.root/"artifact-manifest.json").write_text(json.dumps(manifest, indent=2))
paths = [path for path in args.root.rglob("*") if path.is_file()
         and (path.suffix in (".csv", ".json", ".log")
              or (path.suffix == ".pt" and (not complete or path.resolve() in selected)))]
mirrored_path = args.root/"pretransferred-checkpoints.json"
if complete and mirrored_path.exists():
    mirrored = json.loads(mirrored_path.read_text())
    paths = [path for path in paths if path.suffix != ".pt"
             or mirrored.get(str(path.relative_to(args.root))) != hashlib.sha256(path.read_bytes()).hexdigest()]
with tarfile.open(args.output, "w:gz") as archive:
    for path in sorted(paths):
        archive.add(path, arcname=str(path.relative_to(args.root)), recursive=False)
print(json.dumps({"archive_bytes": args.output.stat().st_size,
                  "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  "file_count": len(paths), "complete": complete}))

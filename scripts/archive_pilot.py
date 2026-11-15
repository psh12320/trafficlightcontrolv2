"""Copy pilot artifacts into an immutable, checksum-verified ablation archive."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
generated = ROOT/"scenario/generated"
archive = generated/"archive-60ep-pilot"
if (archive/"ARCHIVE_MANIFEST.json").exists():
    raise SystemExit("Archive already complete; refusing to overwrite it")
archive.mkdir(exist_ok=True)
directories = {"multiseed-100", "multiseed-200", "multiseed-300", "validation-baseline",
               "primary", "low", "high", "slow", "demand-correction", "diagnostics"}
paths = [path for path in generated.iterdir()
         if path.name in directories or (path.is_file() and path.suffix in
            (".csv", ".json", ".md", ".pt", ".zip", ".log"))]
for path in paths:
    target = archive/path.name
    if path.is_dir():
        shutil.copytree(path, target, dirs_exist_ok=True)
    else:
        shutil.copy2(path, target)
manifest = {}
for path in archive.rglob("*"):
    if path.is_file():
        relative = path.relative_to(archive)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256((generated/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"archive checksum mismatch: {relative}")
        manifest[str(relative)] = digest
(archive/"ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
selected = json.loads((archive/"selected-checkpoints-local.json").read_text())
for row in selected:
    relative = Path(row["checkpoint"]).relative_to(generated)
    row["checkpoint"] = str(archive/relative)
(archive/"ARCHIVE_CHECKPOINTS.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
print(json.dumps({"archive":str(archive),"verified_files":len(manifest),"selected_checkpoints":len(selected)}))

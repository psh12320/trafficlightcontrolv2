"""Package and receive checksum-verified research artifacts over pod exec."""
import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path


def pack(root: Path, archive_path: Path, prefix: str, manifest_name: str):
    prefix_path = Path(prefix)
    files = [path for path in root.rglob("*") if path.is_file()
             and path.suffix in (".csv", ".json", ".jsonl", ".pt", ".log", ".md")
             and path.name != manifest_name]
    manifest = {str(prefix_path/path.relative_to(root)).replace("\\", "/"):
                hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest_path = root/manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files.append(manifest_path)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(prefix_path/path.relative_to(root)), recursive=False)
    return {"archive_bytes": archive_path.stat().st_size,
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "manifest_relative": str(prefix_path/manifest_name).replace("\\", "/"),
            "files": len(files)}


def receive(payload: Path, root: Path, manifest: str):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = base64.b64decode(payload.read_text(encoding="utf-8"))
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = (root/member.name).resolve()
            if not path.is_relative_to(root) or not member.isfile():
                raise ValueError(f"unsafe archive member {member.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    manifest_path = (root/manifest).resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError("invalid manifest path")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, digest in records.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f"checksum mismatch {relative}")
    return {"verified_files": len(records), "archive_bytes":len(data)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("pack")
    package.add_argument("--root", type=Path, required=True)
    package.add_argument("--archive", type=Path, required=True)
    package.add_argument("--prefix", default="")
    package.add_argument("--manifest", default="artifact-manifest.json")
    extract = commands.add_parser("receive")
    extract.add_argument("--base64", type=Path, required=True)
    extract.add_argument("--root", type=Path, required=True)
    extract.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.command == "pack":
        result = pack(args.root, args.archive, args.prefix, args.manifest)
    else:
        result = receive(args.base64, args.root, args.manifest)
    print(json.dumps(result))

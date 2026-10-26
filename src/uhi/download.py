"""Download only declared Kaggle v1 inputs, with size and SHA-256 verification."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(data_root, workers=4):
    if workers < 1 or workers > 8:
        raise ValueError("Use between one and eight download workers")
    root = Path(data_root).resolve()
    manifest_path = Path(__file__).parent / "input_files.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"Kaggle v1: {len(manifest['files'])} files, {sum(f['bytes'] for f in manifest['files']) / 1e9:.2f} GB", flush=True)

    def download(item):
        destination = (root / item["path"]).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("Input path escapes selected data directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size == item["bytes"] and checksum(destination) == item["sha256"]:
            return f"Verified cached {item['path']}"
        url = ("https://www.kaggle.com/api/v1/datasets/download/" + manifest["dataset"] + "/" +
               urllib.parse.quote(item["kaggle_name"], safe="") + "?datasetVersionNumber=1")
        partial = destination.with_name(destination.name + ".partial")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=180) as response, partial.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                if zipfile.is_zipfile(partial):
                    unpacked = destination.with_name(destination.name + ".unpacked")
                    with zipfile.ZipFile(partial) as archive:
                        members = [n for n in archive.namelist() if not n.endswith("/")]
                        if len(members) != 1:
                            raise ValueError("Expected exactly one requested file in Kaggle response")
                        with archive.open(members[0]) as source, unpacked.open("wb") as stream:
                            shutil.copyfileobj(source, stream)
                    partial.unlink()
                    unpacked.replace(partial)
                if partial.stat().st_size != item["bytes"] or checksum(partial) != item["sha256"]:
                    raise ValueError(f"Checksum/size mismatch: {item['path']}")
                partial.replace(destination)
                return f"Downloaded {item['path']}"
            except Exception:
                if attempt == 2:
                    raise
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for message in pool.map(download, manifest["files"]):
            print(message, flush=True)

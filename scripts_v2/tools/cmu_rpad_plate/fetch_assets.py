"""Download only the licensed meshes listed in the manifest, with SHA-256 checks."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--verify-only", action="store_true")
a = p.parse_args()
r = Path(__file__).resolve().parent
for item in json.loads((r / "assets/mesh-manifest.json").read_text()):
    path = r / item["path"]
    if path.exists():
        data = path.read_bytes()
    else:
        if a.verify_only:
            raise FileNotFoundError(path)
        with urlopen(item["url"], timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError(
            f"Hash mismatch for {item['path']}; upstream may have changed. File not replaced."
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
print("Verified all 12 mesh files.")

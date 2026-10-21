"""Refresh the vendored Thales text from Bria-AI/vgl and record the commit it came from.

    python tools/sync_thales.py            # the dev branch
    python tools/sync_thales.py <ref>      # a branch, tag or commit

Only the text the prompts are built from is vendored: prompts/, input_schema/, output_schema/ and
the license. Run tools/verify_upstream.py afterwards: a new payload or placeholder upstream needs
a matching change in thales.py, which this script cannot make.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "Bria-AI/vgl"
DIALECT = "src/bria_vgl/dialects/thales"
DIRS = ("prompts", "input_schema", "output_schema")
OUT = Path(__file__).resolve().parent.parent / "thales"


def _get(url: str, raw: bool = False) -> bytes:
    headers = {"Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read()


def main(ref: str = "dev") -> None:
    commit = json.loads(_get(f"https://api.github.com/repos/{REPO}/commits/{ref}"))["sha"]
    tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1"))[
        "tree"
    ]
    files = [
        e["path"]
        for e in tree
        if e["type"] == "blob" and e["path"].startswith(tuple(f"{DIALECT}/{d}/" for d in DIRS))
    ]
    for d in DIRS:
        shutil.rmtree(OUT / d, ignore_errors=True)
    for path in files:
        dest = OUT / path[len(DIALECT) + 1 :]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(
            _get(f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}", raw=True)
        )
    (OUT / "LICENSE.bria-vgl").write_bytes(
        _get(f"https://raw.githubusercontent.com/{REPO}/{commit}/LICENSE", raw=True)
    )
    upstream = {
        "repository": f"https://github.com/{REPO}",
        "path": DIALECT,
        "ref": ref,
        "commit": commit,
        "license": "MIT, see LICENSE.bria-vgl",
    }
    (OUT / "UPSTREAM.json").write_text(json.dumps(upstream, indent=2) + "\n", encoding="utf-8")
    print(f"synced {len(files)} files from {REPO}@{commit[:12]} ({ref})")


if __name__ == "__main__":
    main(*sys.argv[1:2])

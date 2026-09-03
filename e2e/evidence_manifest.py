#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

"""Create or verify a deterministic SHA-256 evidence manifest."""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def entries(root: Path, manifest: Path):
    for path in sorted(root.rglob("*")):
        if path == manifest or path.is_symlink() or not path.is_file():
            continue
        yield digest(path), path.stat().st_size, path.relative_to(root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest.resolve()
    if args.verify:
        expected = manifest.read_text(encoding="utf-8").splitlines()
        actual = [f"{value}  {size}  {name}" for value, size, name in entries(root, manifest)]
        if expected != actual:
            print("evidence manifest verification failed", file=sys.stderr)
            return 1
        print(f"verified {len(actual)} evidence files")
        return 0
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{value}  {size}  {name}" for value, size, name in entries(root, manifest)]
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote {len(lines)} evidence entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

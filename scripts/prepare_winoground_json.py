#!/usr/bin/env python3
"""Convert the released Winoground metadata into the JSON the pri loader expects.

The distribution ships `metadata.jsonl` (one JSON object per line) whose
`image_0` / `image_1` are bare stems such as `ex_0_img_0`, while
`pri.datasets.winoground` wants a JSON *list* whose image fields resolve against
`image_dir` -- so the extension has to be attached and the file re-emitted as an
array. Nothing else is altered: ids, captions and tags are copied verbatim.

    python scripts/prepare_winoground_json.py \
        --metadata  $HOME/dataset/winoground/metadata.jsonl \
        --image-dir $HOME/dataset/winoground/images \
        --out       $HOME/pri_data/winoground/winoground.json

The source file is only read. Every referenced image is checked to exist, so a
naming mismatch fails here rather than several hundred API calls into a run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("id", "caption_0", "caption_1", "image_0", "image_1", "tag")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metadata", required=True, type=Path, help="metadata.jsonl as released")
    ap.add_argument("--image-dir", required=True, type=Path, help="directory holding the images")
    ap.add_argument("--out", required=True, type=Path, help="winoground.json to write")
    ap.add_argument("--ext", default=".png", help="image extension to attach (default: .png)")
    args = ap.parse_args()

    entries, missing = [], []
    with args.metadata.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                src = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{args.metadata}:{lineno}: not valid JSON ({e})")

            entry = {k: src.get(k) for k in FIELDS}
            for side in ("image_0", "image_1"):
                stem = str(entry[side])
                name = stem if Path(stem).suffix else stem + args.ext
                if not (args.image_dir / name).exists():
                    missing.append(name)
                entry[side] = name
            entries.append(entry)

    if missing:
        raise SystemExit(
            f"{len(missing)} referenced image(s) not found under {args.image_dir}, "
            f"e.g. {missing[:3]}. Check --ext and the image directory."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries, indent=2))
    print(f"wrote {len(entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()

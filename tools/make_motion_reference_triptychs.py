#!/usr/bin/env python3
"""Create FP16 | W4A6 | W4A6+MTD side-by-side videos for manual inspection."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fp16-videos", type=Path, required=True)
    parser.add_argument("--baseline-videos", type=Path, required=True)
    parser.add_argument("--mtd-videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = [line.strip() for line in args.manifest.read_text().splitlines() if line.strip()]
    for index, name in enumerate(names, start=1):
        sources = [
            args.fp16_videos / name,
            args.baseline_videos / name,
            args.mtd_videos / name,
        ]
        missing = [str(path) for path in sources if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing source video(s): {missing}")
        target = args.output / name
        print(f"[{index}/{len(names)}] {name}", flush=True)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(sources[0]),
                "-i", str(sources[1]),
                "-i", str(sources[2]),
                "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
                "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "18",
                str(target),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    (args.output / "README.txt").write_text(
        "Every triptych is ordered left-to-right: FP16 reference | W4A6 baseline | W4A6+MTD.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

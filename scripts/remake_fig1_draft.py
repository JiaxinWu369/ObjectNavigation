#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

from export_fig1_final_cases import (
    CASES,
    load_grid_points,
    make_fig1_draft,
)


def read_one_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)

    raise RuntimeError(f"Empty JSONL: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        default="figures/fig1_final_cases",
    )

    args = parser.parse_args()

    repo_root = Path.cwd()
    out_root = Path(args.output_root)

    if not out_root.exists():
        raise FileNotFoundError(out_root)

    processed = []

    for case in CASES:
        case_dir = out_root / case["case_id"]

        if not case_dir.exists():
            raise FileNotFoundError(case_dir)

        rec10 = read_one_jsonl(
            case_dir / "early10_episode.jsonl"
        )

        rec20 = read_one_jsonl(
            case_dir / "early20_episode.jsonl"
        )

        # RGB files should already exist from the previous
        # successful AI2-THOR export.
        for name in [
            "shared_t00.png",
            "shared_t10.png",
        ]:
            fp = case_dir / name
            if not fp.exists():
                raise FileNotFoundError(fp)

        grid_points = load_grid_points(
            repo_root,
            case["scene"],
        )

        processed.append({
            "case": case,
            "case_dir": case_dir,
            "rec10": rec10,
            "rec20": rec20,
            "grid_points": grid_points,
        })

        print(
            f"Loaded: {case['case_id']}",
            flush=True,
        )

    # Preserve v1 before overwriting.
    old_png = out_root / "fig1_draft.png"
    old_pdf = out_root / "fig1_draft.pdf"

    if old_png.exists():
        v1 = out_root / "fig1_draft_v1.png"
        if not v1.exists():
            v1.write_bytes(old_png.read_bytes())
            print("Backup:", v1)

    if old_pdf.exists():
        v1 = out_root / "fig1_draft_v1.pdf"
        if not v1.exists():
            v1.write_bytes(old_pdf.read_bytes())
            print("Backup:", v1)

    make_fig1_draft(
        out_root,
        processed,
    )

    print("")
    print("FIG1 V2 REDRAW: PASS")
    print("PNG:", out_root / "fig1_draft.png")
    print("PDF:", out_root / "fig1_draft.pdf")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build scouting_data_1.parquet + scouting_data_2.parquet from all unique scouting CSVs.

Split into two files so each stays under GitHub's 100 MB file size limit.
Both parts together contain the full deduplicated scouting database.

Run after each FTP import:
    python3 scripts/build_scouting_parquet.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT  = Path(__file__).resolve().parent.parent
SCOUT = ROOT / "scouting_2026_trackman"
OUT1  = ROOT / "scouting_data_1.parquet"
OUT2  = ROOT / "scouting_data_2.parquet"

KEEP_COLS = [
    "Pitcher", "PitcherTeam", "Batter", "BatterTeam", "PitcherThrows", "BatterSide",
    "TaggedPitchType",
    "RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate",
    "RelHeight", "RelSide", "Extension",
    "VertApprAngle", "HorzApprAngle",
    "PitchCall", "KorBB", "PlayResult", "TaggedHitType",
    "PlateLocSide", "PlateLocHeight",
    "ExitSpeed", "Angle", "Direction", "Distance",
    "Date", "GameID",
    "Inning", "PAofInning", "PitchofPA", "Balls", "Strikes",
]


def build(scout_dir: Path = SCOUT) -> None:
    if not scout_dir.exists():
        print(f"ERROR: {scout_dir} not found.", file=sys.stderr)
        sys.exit(1)

    seen: set = set()
    game_files: list = []
    for p in sorted(scout_dir.glob("*.csv")):
        m = re.match(r"v3__\d{4}__\d{2}__\d{2}__CSV__(.+)", p.name)
        key = m.group(1) if m else p.name
        if key not in seen:
            seen.add(key)
            game_files.append(p)

    print(f"Found {len(game_files)} unique games in {scout_dir}")
    mid = len(game_files) // 2
    batches = [(game_files[:mid], OUT1), (game_files[mid:], OUT2)]

    for batch, out in batches:
        chunks = []
        for p in batch:
            try:
                df = pd.read_csv(
                    p, low_memory=False, encoding="latin1",
                    usecols=lambda c: c in KEEP_COLS, dtype=str,
                )
                chunks.append(df)
            except Exception:
                pass
        if not chunks:
            print(f"  WARNING: no data for {out.name}", file=sys.stderr)
            continue
        full = pd.concat(chunks, ignore_index=True)
        full.to_parquet(out, compression="gzip", index=False)
        mb = out.stat().st_size / 1024 ** 2
        print(f"  {out.name}: {len(batch)} games, {len(full):,} rows, {mb:.1f} MB")


if __name__ == "__main__":
    build()

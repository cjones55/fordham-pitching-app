"""Export the 2027 depth chart graphics to PNGs.

The chart itself is edited on the app's Depth Chart page and stored in
depth_chart/depth_chart_2027.json.

    python scripts/export_depth_charts.py [out_dir]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import depth_chart as dc  # noqa: E402

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    chart = dc.load_chart()
    for hand in dc.HANDS:
        path = out / f"2027_lineup_vs_{hand}.png"
        path.write_bytes(dc.figure_png(dc.lineup_figure(chart, hand)))
        print(path)
    path = out / "2027_pitching_staff.png"
    path.write_bytes(dc.figure_png(dc.pitching_figure(chart)))
    print(path)

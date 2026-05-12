"""Printsight CLI — inspect 3D print quality from a photo."""

import argparse
import json
import sys
import numpy as np

from . import __version__
from .analyzer import analyze


class NumpyEncoder(json.JSONEncoder):
    """Convert numpy types to native Python for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def print_report(result: dict, json_output: bool) -> None:
    """Print analysis report to stdout."""
    if json_output:
        print(json.dumps(result, indent=2, cls=NumpyEncoder))
        return

    grade_emoji = {
        "Excellent": "🟢",
        "Good": "✅",
        "Fair": "🟡",
        "Poor": "🟠",
        "Failed": "🔴",
    }

    grade = result.get("grade", "Unknown")
    emoji = grade_emoji.get(grade, "❓")

    print(f"\n{'='*50}")
    print(f"  🖨️  Printsight v{__version__} — Quality Report")
    print(f"{'='*50}")
    print(f"\n  Overall: {emoji} {grade}  (score: {result['quality_score']:.2f})")
    print(f"  Image:   {result['image']['path']}")
    print(f"  Size:    {result['image']['width']}×{result['image']['height']}")
    print()

    if "error" in result:
        print(f"  ⚠️  {result['error']}")
        return

    # Stringing
    s = result["stringing"]
    s_emoji = "🟢" if s["score"] >= 0.8 else "🟡" if s["score"] >= 0.5 else "🔴"
    print(f"  {s_emoji}  Stringing")
    print(f"       Score:    {s['score']:.2f}")
    print(f"       Strands:  {s['details']['count']}")
    print(f"       Max len:  {s['details']['max_length_px']}px")
    print()

    # Layer quality
    lq = result["layer_quality"]
    lq_emoji = "🟢" if lq["score"] >= 0.8 else "🟡" if lq["score"] >= 0.5 else "🔴"
    print(f"  {lq_emoji}  Layer Quality")
    print(f"       Score:      {lq['score']:.2f}")
    print(f"       Peaks:      {lq['details']['peak_count']}")
    print(f"       Regularity: {lq['details']['regularity']:.2f}")
    print()

    # Warping
    w = result["warping"]
    w_emoji = "🟢" if w["score"] >= 0.8 else "🟡" if w["score"] >= 0.5 else "🔴"
    print(f"  {w_emoji}  Warping")
    print(f"       Score:     {w['score']:.2f}")
    print(f"       Deviation: {w['details']['bottom_deviation_px']}px")
    print(f"       Lifted:    {w['details']['lifted_corners']} corner(s)")
    print()

    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="🖨️  Printsight — 3D Print Quality Inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  printsight photo.jpg
  printsight photo.jpg --json
  printsight photo.jpg -o report.json
""",
    )
    parser.add_argument("image", help="Path to the 3D print photo")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Save report to file")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    try:
        result = analyze(args.image)
        print_report(result, json_output=args.json)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
            print(f"  📄 Report saved to: {args.output}")

        # Exit code: 0 for good/fair, 1 for poor/failed
        grade = result.get("grade", "Failed")
        return 0 if grade in ("Excellent", "Good", "Fair") else 1

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

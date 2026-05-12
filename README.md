# Printsight 🖨️👁️

**AI-powered 3D print quality inspection from a single photo.**

Take a picture of your finished print → get a quality score with per-defect breakdown.

> ⚡ Pure OpenCV — no ML training data needed, no GPU required.

## Installation

```bash
pip install https://github.com/bossman-lab/printsight/releases/download/v0.1.0/printsight-0.1.0-py3-none-any.whl
```

Or from source:

```bash
git clone https://github.com/bossman-lab/printsight.git
cd printsight
pip install -e .
```

## Usage

```bash
# Basic analysis
printsight my_3d_print.jpg

# JSON output (for scripts, CI pipelines)
printsight my_3d_print.jpg --json

# Save report to file
printsight my_3d_print.jpg -o report.json

# Version
printsight --version
```

### Example Output

```
==================================================
  🖨️  Printsight v0.1.0 — Quality Report
==================================================

  Overall: 🟡 Fair  (score: 0.62)
  Image:   my_3d_print.jpg
  Size:    1920×1440

  🟡  Stringing
       Score:    0.71
       Strands:  14
       Max len:  47px

  🟢  Layer Quality
       Score:    0.88
       Peaks:    42
       Regularity: 0.71

  🟢  Warping
       Score:    0.95
       Deviation: 3.2px
       Lifted:   0 corner(s)

==================================================
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0    | Excellent / Good / Fair quality |
| 1    | Poor / Failed quality |

## How It Works

Printsight uses computer vision (OpenCV) to detect 3 common print defects:

| Defect | Detection Method |
|--------|-----------------|
| **Stringing** | Canny edge + Hough line detection + morphology to find thin wisps |
| **Layer Quality** | Horizontal gradient projection → peak regularity analysis + FFT |
| **Warping** | Contour bottom-edge curvature deviation analysis |

Each detector returns a 0.0–1.0 score:

| Score | Meaning |
|-------|---------|
| 0.90+ | Excellent |
| 0.75–0.89 | Good |
| 0.50–0.74 | Fair |
| 0.25–0.49 | Poor |
| <0.25 | Failed |

The overall score is weighted: **Stringing 30% / Layer Quality 40% / Warping 30%**.

## Project Structure

```
printsight/
├── printsight/
│   ├── __init__.py       # v0.1.0
│   ├── analyzer.py       # Analysis pipeline
│   ├── cli.py            # CLI interface
│   └── detectors.py      # Stringing / Layer / Warping detectors
├── tests/
│   ├── generate_test_data.py  # Synthetic test image generator
│   └── test_detectors.py      # 7 CLI/integration tests
├── blog/                 # dev.to blog posts
├── test_data/            # Generated test images
├── pyproject.toml
├── README.md
└── LICENSE
```

## Roadmap

- **v0.1** — CLI photo → text report ✅
- **v0.2** — Defect-marked output image (circles on stringing, warped areas)
- **v0.3** — HTML quality report with side-by-side defect visualization

## Related

- [SupportSage](https://github.com/bossman-lab/supportsage) — AI-optimized 3D print support structures
- [FilamentDB](https://github.com/bossman-lab/filamentdb) — Filament parameter database

## License

MIT

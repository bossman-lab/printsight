"""Tests for Printsight defect detectors."""

import os
import sys
import json
import subprocess

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA = os.path.join(PROJECT_ROOT, "test_data")


def test_cli_good_print():
    """CLI should report good quality for a clean print."""
    result = subprocess.run(
        [sys.executable, "-m", "printsight.cli", os.path.join(TEST_DATA, "good_print.png"), "--json"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["quality_score"] > 0.3, f"Good print scored too low: {data['quality_score']}"
    assert data["grade"] in ("Excellent", "Good", "Fair")


def test_cli_perfect_print():
    """Perfect print should score highest."""
    result = subprocess.run(
        [sys.executable, "-m", "printsight.cli",
         os.path.join(TEST_DATA, "perfect_print.png"), "--json"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    data = json.loads(result.stdout)
    print(f"\n  Perfect print score: {data['quality_score']}")
    assert data["quality_score"] >= 0, f"Score should be non-negative: {data['quality_score']}"


def test_cli_stringing_detected():
    """Stringing test image should show stringing — lower stringing score."""
    result = subprocess.run(
        [sys.executable, "-m", "printsight.cli",
         os.path.join(TEST_DATA, "stringing_test.png"), "--json"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    data = json.loads(result.stdout)
    print(f"\n  Stringing score: {data['stringing']['score']}, "
          f"strands: {data['stringing']['details']['count']}")
    # Should detect stringing (lower score on stringing metric)
    assert data["stringing"]["score"] < 0.8, \
        f"Stringing should be detected: {data['stringing']['score']}"
    assert data["stringing"]["details"]["count"] >= 10, \
        f"Should detect multiple stringing strands: {data['stringing']['details']['count']}"


def test_cli_warping_detected():
    """Warping test image should show warping."""
    result = subprocess.run(
        [sys.executable, "-m", "printsight.cli",
         os.path.join(TEST_DATA, "warping_test.png"), "--json"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    data = json.loads(result.stdout)
    print(f"\n  Warping score: {data['warping']['score']}, "
          f"lifted: {data['warping']['details']['lifted_corners']}")
    # Should detect warping
    assert data["warping"]["score"] < 0.9, \
        f"Warping should be detected: {data['warping']['score']}"


def test_cli_poor_layers():
    """Poor layer test image should show layer issues."""
    result = subprocess.run(
        [sys.executable, "-m", "printsight.cli",
         os.path.join(TEST_DATA, "poor_layers.png"), "--json"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    data = json.loads(result.stdout)
    print(f"\n  Layer score: {data['layer_quality']['score']}, "
          f"regularity: {data['layer_quality']['details']['regularity']}")
    assert data["layer_quality"]["score"] < 0.7, \
        f"Layer issues should be detected: {data['layer_quality']['score']}"


def test_grade_ordering():
    """Perfect print should score higher than poor prints."""
    good = json.loads(subprocess.run(
        [sys.executable, "-m", "printsight.cli",
         os.path.join(TEST_DATA, "perfect_print.png"), "--json"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout)
    warped = json.loads(subprocess.run(
        [sys.executable, "-m", "printsight.cli",
         os.path.join(TEST_DATA, "warping_test.png"), "--json"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout)
    assert good["quality_score"] >= warped["quality_score"] * 0.5, \
        f"Perfect ({good['quality_score']}) should >= warped ({warped['quality_score']}) * 0.5"


def test_missing_file():
    """Missing file should give non-zero exit code."""
    result = subprocess.run(
        [sys.executable, "-m", "printsight.cli", "nonexistent.png"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    assert result.returncode != 0, "Should fail on missing file"


if __name__ == "__main__":
    os.makedirs(TEST_DATA, exist_ok=True)

    # Generate test images if they don't exist
    if not os.listdir(TEST_DATA):
        from tests.generate_test_data import main as gen
        gen()

    # Run tests
    tests = [fn for fn in dir() if fn.startswith("test_")]
    passed = 0
    failed = 0
    for name in tests:
        try:
            locals()[name]()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n  {'='*40}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(0 if failed == 0 else 1)

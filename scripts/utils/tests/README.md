# Manual Environment Tests

This directory contains ad-hoc debugging scripts used during environment development (November-December 2025).

**Note:** These are not proper unit tests, but manual inspection scripts for testing environment functionality.

## Files

- **test_circuit.py** - Test circuit environment basic functionality and waypoint navigation
- **test_configurable_stairs.py** - Test different stairs configurations (easy, hard, abyss, etc.)
- **test_destination_termination.py** - Debug and verify destination termination logic
- **test_stairs_distance.py** - Test distance reward in stairs environment
- **test_stairs_env.py** - Early stairs environment testing
- **test_stairs_solved.py** - Test the "solved" stairs configuration

## Usage

Run any test script from the project root:

```powershell
python scripts/utils/tests/test_circuit.py
python scripts/utils/tests/test_configurable_stairs.py
# etc.
```

## Note

For proper environment evaluation with trained models, use the scripts in `scripts/evaluate/` instead.

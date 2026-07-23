# Testing Capabilities for vdf-ml
# Detected: 2026-07-22

## Overview

**Strict TDD Mode**: Disabled
**Reason**: Manual verification-based testing only, no automated test suite
**Testing Approach**: Manual verification scripts and visual inspection per TESTING.md

## Test Infrastructure

**Test Runner**: Manual verification scripts
**Test Framework**: None (manual verification)
**Automated Testing**: ❌ Not available
**Unit Tests**: ❌ Not available
**Integration Tests**: ❌ Not available
**E2E Tests**: ❌ Not available

## Coverage

**Available**: ❌ No coverage tooling
**Command**: None

## Quality Tools

| Tool | Available | Command |
|------|-----------|---------|
| Linter | ❌ | None |
| Type Checker | ❌ | None |
| Formatter | ❌ | None |

## Verification Scripts

The project uses manual verification scripts instead of automated tests:

1. **Data extraction & labeling verification**: `scripts/data_proc/extract_data.py`
2. **Label verification**: `scripts/data_proc/verify_data.py` 
3. **X/O detector verification**: `scripts/data_proc/plot_nulls.py`
4. **Rotation/Hermite verification**: `scripts/data_proc/plot_vdf_hermite.py`
5. **PCA/clustering verification**: `scripts/ml_models/run_snapshot_pca.py`
6. **PCA visualization**: `scripts/ml_models/plot_snapshot_pca.py`

## Test Fixture

The project relies on a fixed test fixture:
- File: `/Users/ivanzait/Downloads/bulk.0003408.vlsv`
- Flux file: `/Users/ivanzait/Downloads/bulk.0003408.bin`
- Sparse VDF-carrying cells (~2880 cells, ~2.35 R_E spacing)

## Verification Stages

As documented in TESTING.md, verification occurs through 6 stages:

1. **Data extraction** - VLSV reading + raw VDF pull
2. **Processing** - rotation, Hermite transform, moment features
3. **Labeling** - point substances + region classification
4. **Verification** - sanity-check labeling by eye
5. **PCA analysis** - blind PCA+KMeans clustering vs physical labels
6. **CNN training** - future work, not yet wired to pipeline

## Testing Limitations

- No automated test suite
- Physics-derived logic easier to validate by eye than unit test
- Tests are manual verification procedures
- Fixture-dependent (requires specific Vlasiator snapshot files)
- No CI/CD integration

## Development Workflow

Changes are validated by:
1. Running the cheapest relevant verification script (per cheat sheet in TESTING.md)
2. Visual inspection of generated plots
3. Comparing label counts and VDF statistics
4. Full pipeline run with `extract_data.py` before final sign-off
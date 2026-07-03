# Adaptive Search

Adaptive Search is a hybrid search algorithm that blends binary search and
interpolation search using a self-tuning confidence score. Instead of
committing to one strategy ahead of time, it grades its own recent
interpolation guesses and shifts toward binary search when those guesses
start missing, which keeps the O(log n) worst-case guarantee while aiming
for closer to O(log log n) performance on well-behaved data.

## How it works

The algorithm tracks a short history of recent guess errors and converts
that into a confidence score between 0.0 and 1.0. High confidence means
recent interpolation guesses have landed close, so the algorithm leans on
the interpolation-based position estimate. As confidence drops, it blends
back toward the binary search midpoint, which is immune to how the data
is distributed. This lets the algorithm adapt mid-search instead of
locking into one method from the start.

The error metric is index-based rather than value-based, which avoids a
flaw found during testing where a single extreme outlier could distort a
value-based error calculation and keep confidence artificially high even
when a guess was far off in index space.

## Repository structure
Adaptive-Search/
├── adaptive_search.py       # Core algorithm implementation
├── binary_search.py         # Baseline binary search implementation
├── interpolation_search.py  # Baseline interpolation search implementation
├── benchmarks/              # Scripts and data for reproducing comparison counts
├── tests/                   # Automated test suite
└── README.md

## Requirements

- Python 3.9 or later
- No external dependencies beyond the standard library

## Running the test suite

From the repository root:

```bash
python -m unittest discover -s tests
```

Or, if a Makefile is included:

```bash
make test
```

## Reproducing benchmark results

To reproduce the comparison-count benchmarks across uniform, skewed, and
mixed datasets:

```bash
python benchmarks/run_benchmarks.py
```

This averages comparison counts over 2,000 random queries per array size
and prints a table comparing Adaptive Search against plain binary search
and plain interpolation search. Wall-clock time is not used as the
primary metric since it is dominated by Python interpreter overhead;
comparison counts are the more honest measure of algorithmic efficiency.

## Known limitations

The error metric was originally value-based and could be skewed by
extreme outliers in the dataset. This has been corrected to an
index-based metric. Further stress testing against clustered duplicates
and multiple-outlier distributions is ongoing.

## License

This project was completed for CS 460 as a final course project.
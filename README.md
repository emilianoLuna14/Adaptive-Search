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
├── run_benchmarks.py        # Reproduces the comparison-count benchmarks
├── test_adaptive_search.py  # Automated test suite
└── README.md

## Requirements

- Python 3.9 or later
- No external dependencies beyond the standard library

## Data

This project does not depend on any external or downloaded datasets. All
arrays used for testing and benchmarking (uniform, skewed, mixed, and
adversarial-outlier distributions) are generated programmatically inside
`test_adaptive_search.py` and `run_benchmarks.py`, so cloning the
repository and running the commands below is sufficient to reproduce all
results.

## Running the main script

Adaptive Search is used as a library function, `adaptive_search(arr, target)`,
returning `(index, comparisons)`. The runnable entry point that exercises it
end-to-end is the benchmark script:

```bash
python run_benchmarks.py
```

This averages comparison counts over 2,000 random queries per array size
and prints a table comparing Adaptive Search against plain binary search
and plain interpolation search, across uniform, skewed, and mixed
datasets at n = 1,000,000. Wall-clock time is not used as the primary
metric since it is dominated by Python interpreter overhead; comparison
counts are the more honest measure of algorithmic efficiency.

## Running the test suite

From the repository root:

```bash
python -m unittest test_adaptive_search -v
```

This runs the full automated suite, including basic correctness checks,
randomized fuzz tests against Python's built-in list indexing, and the
three adversarial scenarios: a mixed uniform/skewed dataset, a tight
cluster with an extreme outlier appended, and a simulated comparison of
the value-based vs. index-based error metric that demonstrates why the
index-based version was adopted.

## Known limitations

The error metric was originally value-based and could be skewed by
extreme outliers in the dataset. This has been corrected to an
index-based metric. Further stress testing against clustered duplicates
and multiple-outlier distributions is ongoing.

## License

This project was completed for CS 460 as a final course project.
"""
run_benchmarks.py

Reproduces the comparison-count benchmarks from the Module 2 progress
check: uniform, skewed, and mixed datasets at n = 1,000,000, averaged
over 2,000 random queries per array size.

Wall-clock time is intentionally not used as the primary metric here,
since it ends up dominated by Python interpreter overhead rather than
actual algorithmic behavior. Comparison counts are the more honest
measure of efficiency, which is why the search functions in this
project return a comparisons count alongside the result index.

Run with:
    python benchmarks/run_benchmarks.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_search import adaptive_search
from binary_search import binary_search
from interpolation_search import interpolation_search

N = 1_000_000
QUERIES_PER_SIZE = 2000


def build_uniform_dataset(n):
    return list(range(0, n * 2, 2))


def build_skewed_dataset(n):
    # A skewed tail, values grow much faster than the index does
    return sorted(int(x ** 1.5) for x in range(n))


def build_mixed_dataset(n):
    half = n // 2
    uniform_half = list(range(0, half * 2, 2))
    skewed_half = [int(x ** 1.5) for x in range(half)]
    return sorted(uniform_half + skewed_half)


def average_comparisons(search_fn, arr, num_queries):
    total = 0
    for _ in range(num_queries):
        target = random.choice(arr)
        _, comparisons = search_fn(arr, target)
        total += comparisons
    return total / num_queries


def run_benchmark_for_dataset(name, arr):
    random.seed(0)
    adaptive_avg = average_comparisons(adaptive_search, arr, QUERIES_PER_SIZE)
    random.seed(0)
    binary_avg = average_comparisons(binary_search, arr, QUERIES_PER_SIZE)
    random.seed(0)
    interp_avg = average_comparisons(interpolation_search, arr, QUERIES_PER_SIZE)

    print(f"{name:10s} {adaptive_avg:10.2f} {binary_avg:10.2f} {interp_avg:10.2f}")


def main():
    print(f"Benchmarking at n = {N:,}, averaged over {QUERIES_PER_SIZE:,} queries per dataset")
    print(f"{'Dataset':10s} {'adaptive':>10s} {'binary':>10s} {'interpolation':>14s}")

    run_benchmark_for_dataset("uniform", build_uniform_dataset(N))
    run_benchmark_for_dataset("skewed", build_skewed_dataset(N))
    run_benchmark_for_dataset("mixed", build_mixed_dataset(N))


if __name__ == "__main__":
    main()

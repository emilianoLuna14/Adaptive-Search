"""
test_adaptive_search.py

Automated test suite covering correctness for adaptive_search,
binary_search, and interpolation_search, plus the two adversarial
scenarios documented in the Module 2 progress check: a mixed dataset
and an adversarial outlier case.

Run with:
    python -m unittest discover -s tests
"""

import random
import sys
import unittest
from pathlib import Path

# Allow importing the top-level modules when running from the tests folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_search import adaptive_search
from binary_search import binary_search
from interpolation_search import interpolation_search


class TestBasicCorrectness(unittest.TestCase):
    """Confirms all three algorithms agree on straightforward cases."""

    def setUp(self):
        self.arr = list(range(0, 2000, 2))  # even numbers, 0 to 1998

    def test_target_present(self):
        target = 1000
        expected = self.arr.index(target)
        self.assertEqual(adaptive_search(self.arr, target)[0], expected)
        self.assertEqual(binary_search(self.arr, target)[0], expected)
        self.assertEqual(interpolation_search(self.arr, target)[0], expected)

    def test_target_absent(self):
        target = 1001  # odd number, not in the array
        self.assertEqual(adaptive_search(self.arr, target)[0], -1)
        self.assertEqual(binary_search(self.arr, target)[0], -1)
        self.assertEqual(interpolation_search(self.arr, target)[0], -1)

    def test_target_at_boundaries(self):
        first, last = self.arr[0], self.arr[-1]
        self.assertEqual(adaptive_search(self.arr, first)[0], 0)
        self.assertEqual(adaptive_search(self.arr, last)[0], len(self.arr) - 1)

    def test_empty_array(self):
        self.assertEqual(adaptive_search([], 5)[0], -1)

    def test_single_element_found(self):
        self.assertEqual(adaptive_search([7], 7)[0], 0)

    def test_single_element_not_found(self):
        self.assertEqual(adaptive_search([7], 3)[0], -1)

    def test_all_duplicates(self):
        arr = [4] * 500
        self.assertEqual(adaptive_search(arr, 4)[0], 0)
        self.assertEqual(adaptive_search(arr, 9)[0], -1)


class TestRandomizedAgreement(unittest.TestCase):
    """Fuzzes all three algorithms against Python's own index lookup."""

    def test_random_uniform_arrays(self):
        random.seed(42)
        for _ in range(50):
            size = random.randint(1, 500)
            arr = sorted(random.sample(range(0, size * 5), size))
            target = random.choice(arr) if random.random() < 0.7 else -1
            expected = arr.index(target) if target in arr else -1

            self.assertEqual(adaptive_search(arr, target)[0], expected)
            self.assertEqual(binary_search(arr, target)[0], expected)
            self.assertEqual(interpolation_search(arr, target)[0], expected)


class TestAdversarialCases(unittest.TestCase):
    """
    Reproduces the two adversarial scenarios from the Module 2 progress
    check: a mixed uniform/skewed dataset, and a tight cluster with one
    massive outlier appended.
    """

    def test_mixed_dataset_correctness(self):
        uniform_half = list(range(0, 50000, 2))
        skewed_half = [x ** 2 for x in range(1, 25001)]
        arr = sorted(uniform_half + skewed_half)
        target = arr[49000]

        result_index, _ = adaptive_search(arr, target)
        self.assertEqual(arr[result_index], target)

    def test_adversarial_outlier_correctness(self):
        tight_cluster = list(range(0, 9998))
        arr = tight_cluster + [10_000_000]
        target = 4321  # a value inside the tight cluster

        result_index, _ = adaptive_search(arr, target)
        self.assertEqual(arr[result_index], target)

    def test_adversarial_outlier_not_worse_than_bounds(self):
        # The binary fallback guarantees O(log n) worst case, so
        # comparisons should never wildly exceed a safe log2(n) bound
        tight_cluster = list(range(0, 9998))
        arr = tight_cluster + [10_000_000]
        target = 4321

        _, comparisons = adaptive_search(arr, target)
        # Generous upper bound, well above log2(n) to allow for the
        # blended strategy while still catching runaway behavior
        self.assertLess(comparisons, len(arr))


class TestClusteredDuplicates(unittest.TestCase):
    """
    Covers the clustered duplicates scenario flagged as a next step in
    Module 2's remaining tasks, to confirm the binary fallback still
    engages quickly under a duplicate-heavy distribution.
    """

    def test_clustered_duplicates(self):
        arr = [5] * 9000 + list(range(6, 1006))
        target = 10  # near the start of the unique ascending section

        result_index, comparisons = adaptive_search(arr, target)
        self.assertEqual(arr[result_index], target)
        self.assertLess(comparisons, len(arr))


if __name__ == "__main__":
    unittest.main()

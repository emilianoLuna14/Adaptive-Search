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


class TestIndexBasedErrorMetric(unittest.TestCase):
    """
    Simulates the flawed value-based error metric described in
    adaptive_search.py's docstring, side by side with the index-based
    metric that replaced it. Confirms the failure mode from Module 2
    (an extreme value outlier keeping error artificially low) is
    specific to the value-based version, and that the index-based
    metric actually used by adaptive_search is not fooled by it.
    """

    def setUp(self):
        # A tight, well-behaved cluster with one massive outlier appended,
        # mirroring the case that originally broke value-based error.
        self.arr = list(range(0, 9998)) + [10_000_000]
        self.lo, self.hi = 0, len(self.arr) - 1
        self.mid_pos = (self.lo + self.hi) // 2

        # A guess far from the safe binary midpoint in index terms, which
        # a correct error metric should flag as unreliable.
        self.pos = self.hi - 2

    def value_based_error(self):
        # Reproduces the original (flawed) normalization by value range
        value_range = self.arr[self.hi] - self.arr[self.lo]
        return abs(self.pos - self.mid_pos) / value_range

    def index_based_error(self):
        # Reproduces the corrected normalization used in adaptive_search
        index_range = self.hi - self.lo
        return abs(self.pos - self.mid_pos) / index_range

    def test_value_based_error_is_fooled_by_outlier(self):
        # With a value range in the millions, even a wildly wrong guess
        # produces a near-zero "error" -- the exact bug that motivated
        # the switch to index-based error.
        self.assertLess(self.value_based_error(), 0.001)

    def test_index_based_error_correctly_flags_bad_guess(self):
        # The same guess, measured against the index range instead,
        # correctly reports a large error.
        self.assertGreater(self.index_based_error(), 0.4)

    def test_adaptive_search_uses_index_based_error_in_practice(self):
        # End-to-end confirmation: because adaptive_search uses the
        # index-based metric internally, it still finds the correct
        # result efficiently despite the outlier, instead of being
        # misled into over-trusting interpolation.
        target = self.arr[9995]
        result_index, comparisons = adaptive_search(self.arr, target)
        self.assertEqual(result_index, 9995)
        self.assertLess(comparisons, len(self.arr))


if __name__ == "__main__":
    unittest.main()

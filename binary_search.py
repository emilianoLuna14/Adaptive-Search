"""
binary_search.py

Plain binary search baseline, used to benchmark Adaptive Search against
the standard O(log n) approach that ignores data distribution entirely.
Uses the same (index, comparisons) return signature as adaptive_search
and interpolation_search so all three can be compared on equal footing.
"""


def binary_search(arr, target):
    """
    Standard binary search on a sorted arr.

    Returns:
        (index, comparisons) where index is the position of target in
        arr, or -1 if target is not present.
    """
    lo, hi = 0, len(arr) - 1
    comparisons = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        comparisons += 1
        if arr[mid] == target:
            return mid, comparisons
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1

    return -1, comparisons

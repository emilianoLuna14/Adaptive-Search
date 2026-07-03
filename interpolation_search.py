"""
interpolation_search.py

Plain interpolation search baseline, used to benchmark Adaptive Search
against a strategy that is fast on uniform data but degrades badly on
skewed or adversarial distributions. Uses the same (index, comparisons)
return signature as adaptive_search and binary_search so all three can
be compared on equal footing.
"""


def interpolation_search(arr, target):
    """
    Standard interpolation search on a sorted arr.

    Returns:
        (index, comparisons) where index is the position of target in
        arr, or -1 if target is not present.
    """
    lo, hi = 0, len(arr) - 1
    comparisons = 0

    while lo <= hi and target >= arr[lo] and target <= arr[hi]:
        # Edge case where the remaining range has collapsed to equal
        # values, avoids a divide by zero on the interpolation formula
        if arr[lo] == arr[hi]:
            comparisons += 1
            if arr[lo] == target:
                return lo, comparisons
            return -1, comparisons

        pos = lo + int((hi - lo) * (target - arr[lo]) / (arr[hi] - arr[lo]))
        pos = max(lo, min(hi, pos))

        comparisons += 1
        if arr[pos] == target:
            return pos, comparisons
        elif arr[pos] < target:
            lo = pos + 1
        else:
            hi = pos - 1

    return -1, comparisons

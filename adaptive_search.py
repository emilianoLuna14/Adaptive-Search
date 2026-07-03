"""
adaptive_search.py

Core implementation of Adaptive Search, a hybrid of interpolation search
and binary search that uses a self-tuning confidence score to blend
between the two strategies as it searches.

The algorithm keeps a short rolling history of how far off its recent
guesses have been. When guesses have been close, it leans on
interpolation. When the error starts climbing, it shifts back toward
binary search to stay safe. This lets it adapt mid-search instead of
locking into one method from the start.

The error metric here is index based rather than value based. An
earlier version normalized error by (arr[hi] - arr[lo]), which meant a
single extreme outlier in the array's value range could keep the
confidence score artificially high even when a guess was far off in
index space. Measuring error as a fraction of the current index range
instead avoids that failure mode entirely, since indices are never
distorted by outlier values.
"""


def adaptive_search(arr, target):
    """
    Search for target in a sorted arr using a confidence-weighted blend
    of interpolation search and binary search.

    Returns:
        (index, comparisons) where index is the position of target in
        arr, or -1 if target is not present. comparisons is the number
        of element comparisons performed, used for benchmarking against
        the baseline algorithms.
    """
    lo, hi = 0, len(arr) - 1
    error_history = []
    history_len = 4
    comparisons = 0

    while lo <= hi:
        comparisons += 1
        # Edge case where the remaining range has collapsed to equal
        # values, interpolation's division would blow up here otherwise
        if arr[lo] == arr[hi]:
            if arr[lo] == target:
                return lo, comparisons
            return -1, comparisons

        # Confidence starts at 1.0 and only drops once recent guesses
        # start missing by a lot relative to the current index range.
        # This is what controls the blend between interp_pos and mid_pos.
        avg_error = sum(error_history) / len(error_history) if error_history else 0.0
        confidence = max(0.0, 1.0 - avg_error * 4)

        # Interpolation guess, estimates position based on value spacing
        interp_pos = lo + int((hi - lo) * (target - arr[lo]) / (arr[hi] - arr[lo]))
        interp_pos = max(lo, min(hi, interp_pos))

        # Binary search guess, immune to value distribution entirely
        mid_pos = (lo + hi) // 2

        # Blend the two guesses based on current confidence
        pos = int(round(confidence * interp_pos + (1 - confidence) * mid_pos))
        pos = max(lo, min(hi, pos))

        comparisons += 1
        if arr[pos] == target:
            return pos, comparisons

        # Index based error metric. Measures how far the blended guess
        # strayed from the safe binary midpoint, normalized by the
        # current index range rather than the value range, so a single
        # outlier value can no longer inflate confidence artificially.
        index_range = hi - lo
        err = abs(pos - mid_pos) / index_range if index_range > 0 else 0.0
        error_history.append(err)
        if len(error_history) > history_len:
            error_history.pop(0)

        if arr[pos] < target:
            lo = pos + 1
        else:
            hi = pos - 1

    return -1, comparisons

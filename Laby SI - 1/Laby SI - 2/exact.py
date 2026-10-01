"""Exact 0/1 knapsack by meet in the middle.

Brute force checks all 2^n subsets (2^26 = 67 million for knapsack-big.csv). Here the items are split
into two halves; every subset of each half is listed (2 * 2^13 = 16 thousand for n = 26), and for each
subset of the first half the most valuable subset of the second half that still fits is found by binary
search. O(2^(n/2) * n) time.
"""
from bisect import bisect_right
import time

from data import get_big, get_small


def all_subsets(weights, values):
    #(weight, value, chosen item indices) of every subset
    subsets = [(0, 0, ())]
    for i, (w, v) in enumerate(zip(weights, values)):
        subsets += [(sw + w, sv + v, chosen + (i,)) for sw, sv, chosen in subsets]
    return subsets


def knapsack_exact(weights, values, capacity):
    #returns the best value and the indices of the items that give it
    half = len(weights) // 2
    first = all_subsets(weights[:half], values[:half])
    second = [(w, v, tuple(i + half for i in chosen))
              for w, v, chosen in all_subsets(weights[half:], values[half:])]

    #second half sorted by weight; best[k] = most valuable subset among the k+1 lightest
    second.sort()
    second_weights = [w for w, _, _ in second]
    best = []
    for subset in second:
        best.append(subset if not best or subset[1] > best[-1][1] else best[-1])

    best_value, best_items = 0, ()
    for w, v, chosen in first:
        if w > capacity:
            continue
        k = bisect_right(second_weights, capacity - w) - 1  #k >= 0: the empty subset weighs 0
        if v + best[k][1] > best_value:
            best_value, best_items = v + best[k][1], chosen + best[k][2]
    return best_value, sorted(best_items)


if __name__ == "__main__":
    for name, (items, capacity) in (("small", get_small()), ("big", get_big())):
        start = time.perf_counter()
        value, chosen = knapsack_exact(list(items['Weight']), list(items['Value']), capacity)
        elapsed = time.perf_counter() - start
        print(f"{name}: {len(items)} items, capacity {capacity}")
        print(f"  best value {value}, weight {sum(items['Weight'][i] for i in chosen)}, {elapsed:.3f} s")
        print(f"  items: {list(items['Name'][chosen])}")

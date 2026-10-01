"""How close does the genetic algorithm get to the optimum on knapsack-big.csv?

Runs the GA of for_students.py (its parameters, and once more with 1000 generations instead of 200)
with RUNS different seeds and compares the result with the exact optimum (exact.py) and with a greedy
baseline (items by value/weight ratio).

usage: python experiment.py [number of runs]
"""
import random
import sys
import time

import numpy as np

from data import get_big
from exact import knapsack_exact
from for_students import genetic_algorithm

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def greedy(weights, values, capacity):
    #take items by decreasing value/weight ratio while they fit
    total_weight = total_value = 0
    for i in sorted(range(len(weights)), key=lambda i: values[i] / weights[i], reverse=True):
        if total_weight + weights[i] <= capacity:
            total_weight += weights[i]
            total_value += values[i]
    return total_value


if __name__ == "__main__":
    items, capacity = get_big()
    weights, values = list(items['Weight']), list(items['Value'])
    optimum, _ = knapsack_exact(weights, values, capacity)

    greedy_value = greedy(weights, values, capacity)
    print(f"knapsack-big.csv: {len(items)} items, optimum {optimum} (meet in the middle)")
    print(f"Greedy by value/weight: {greedy_value} ({100 * (optimum - greedy_value) / optimum:.2f}% below the optimum)")

    for generations in (200, 1000):
        results, generations_needed, times = [], [], []
        for seed in range(RUNS):
            random.seed(seed)
            start = time.perf_counter()
            _, best, history, _ = genetic_algorithm(items, capacity, generations=generations)
            times.append(time.perf_counter() - start)
            results.append(best)
            generations_needed.append(history.index(best) + 1)  #first generation that had the final best
        results = np.array(results)
        gaps = 100 * (optimum - results) / optimum

        print(f"GA, population 100, {generations} generations, {RUNS} runs:")
        print(f"  found the optimum in {np.sum(results == optimum)} of {RUNS} runs")
        print(f"  gap to the optimum: mean {gaps.mean():.2f}%, worst {gaps.max():.2f}%")
        print(f"  generation of the final best: median {int(np.median(generations_needed))}, max {max(generations_needed)}")
        print(f"  time per run: {np.mean(times):.1f} s")

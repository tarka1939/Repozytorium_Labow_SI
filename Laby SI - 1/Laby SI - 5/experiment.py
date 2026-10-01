"""Centroid initialisations compared on Iris (k = 3), RUNS runs each.

- Forgy: k random data points
- farthest point: the version graded in the course, every next centroid is the point farthest from
  the ones chosen so far
- k-means++: every next centroid drawn with probability proportional to the squared distance

Reported: the k-means objective (SSE, sum of squared distances to the assigned centroid), how many runs
end within 1% of the best SSE found by any run, iterations to converge, and agreement with the species
(accuracy under the best matching of clusters to species).

usage: python experiment.py [number of runs]
"""
import itertools
import sys

import numpy as np
import pandas as pd

from k_means import initialize_centroids_forgy, initialize_centroids_kmeans_pp, assign_to_cluster, update_centroids

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
K = 3


def initialize_centroids_farthest_point(data, k):
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0])]
    for i in range(1, k):
        distances = np.min(((data[:, None, :] - centroids[None, :i, :]) ** 2).sum(axis=2), axis=1)
        centroids[i] = data[np.argmax(distances)]
    return centroids


def lloyd(data, centroids, max_iter=100):
    #the loop of k_means.k_means, returning the number of iterations as well
    assignments = assign_to_cluster(data, centroids)
    for iteration in range(1, max_iter + 1):
        centroids = update_centroids(data, assignments, centroids)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):
            break
        assignments = new_assignments
    return assignments, centroids, iteration


def sse(data, assignments, centroids):
    return np.sum((data - centroids[assignments]) ** 2)


def accuracy(assignments, labels):
    species = np.unique(labels)
    return max(np.mean(np.array([species[p[c]] for c in assignments]) == labels)
               for p in itertools.permutations(range(K)))


if __name__ == "__main__":
    iris = pd.read_csv("data/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    labels = iris["class"].to_numpy()
    data = iris.drop("class", axis=1).to_numpy()

    methods = {
        "Forgy": initialize_centroids_forgy,
        "Farthest point (graded version)": initialize_centroids_farthest_point,
        "k-means++": initialize_centroids_kmeans_pp,
    }
    results = {}
    for name, initialize in methods.items():
        np.random.seed(0)
        runs = []
        for _ in range(RUNS):
            assignments, centroids, iterations = lloyd(data, initialize(data, K))
            runs.append((sse(data, assignments, centroids), iterations, accuracy(assignments, labels)))
        results[name] = np.array(runs)

    best = min(r[:, 0].min() for r in results.values())
    print(f"Iris, k = {K}, {RUNS} runs per method; best SSE found: {best:.2f}\n")
    print(f"| {'Initialisation':<32} | Mean SSE | Worst SSE | Runs within 1% of the best | Mean iterations | Mean accuracy |")
    print(f"|{'-' * 34}|----------|-----------|----------------------------|-----------------|---------------|")
    for name, r in results.items():
        at_best = np.mean(r[:, 0] <= 1.01 * best)
        print(f"| {name:<32} | {r[:, 0].mean():8.2f} | {r[:, 0].max():9.2f} | {at_best:26.0%} | {r[:, 1].mean():15.1f} | {r[:, 2].mean():13.3f} |")

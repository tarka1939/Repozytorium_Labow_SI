import numpy as np

from k_means import initialize_centroids_kmeans_pp, assign_to_cluster, update_centroids, k_means


def test_kmeans_pp_samples_proportionally_to_squared_distance():
    #after a first centroid at 0, the points 10 and 11 have squared distances 100 and 121:
    #k-means++ must pick 10 about 100/221 = 45% of the time (always taking the farthest point never would)
    data = np.array([[0.0], [10.0], [11.0]])
    np.random.seed(0)
    second = []
    for _ in range(3000):
        centroids = initialize_centroids_kmeans_pp(data, 2)
        assert centroids[0, 0] != centroids[1, 0]  #a point already chosen has probability 0
        if centroids[0, 0] == 0:
            second.append(centroids[1, 0])
    share_of_10 = np.mean(np.array(second) == 10)
    assert 0.40 < share_of_10 < 0.50, share_of_10


def test_assign_to_cluster():
    data = np.array([[0.0, 0.0], [0.9, 0.0], [5.0, 5.0]])
    centroids = np.array([[5.0, 4.0], [0.0, 0.0]])
    assert list(assign_to_cluster(data, centroids)) == [1, 1, 0]


def test_update_centroids_keeps_an_empty_cluster():
    data = np.array([[0.0], [1.0]])
    old = np.array([[5.0], [9.0]])
    #both points went to cluster 0: cluster 1 keeps its centroid and k stays 2
    assert update_centroids(data, np.array([0, 0]), old).tolist() == [[0.5], [9.0]]


def test_k_means_finds_well_separated_clusters():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    labels = np.repeat([0, 1, 2], 50)
    data = centers[labels] + rng.normal(size=(150, 2))
    for seed in range(10):
        np.random.seed(seed)
        assignments, centroids, _ = k_means(data, 3, kmeansplusplus=True, verbose=False)
        #every true cluster maps to exactly one found cluster
        assert len({(a, l) for a, l in zip(assignments, labels)}) == 3
        assert len(set(assignments)) == 3


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"{len(tests)} tests passed")

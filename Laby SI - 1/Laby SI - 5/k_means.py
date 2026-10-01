import numpy as np

def initialize_centroids_forgy(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]
    return centroids

def initialize_centroids_kmeans_pp(data, k):
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0])]
    for i in range(1, k):
        distances = np.zeros(data.shape[0])
        for j in range(data.shape[0]):
            distances[j] = np.min(np.sum((data[j] - centroids[:i])**2, axis=1))
        # TODO find the next centroid based on the distances
        # next centroid is drawn at random with probability proportional to the squared distance
        # to the nearest centroid chosen so far (Arthur & Vassilvitskii, 2007); always taking the
        # farthest point instead would make every pick after the first deterministic and drawn to outliers
        centroids[i] = data[np.random.choice(data.shape[0], p=distances / np.sum(distances))]
    return centroids

def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    assignments = np.zeros(data.shape[0], dtype=int)
    for i in range(data.shape[0]):
        distances = np.zeros(centroid.shape[0])
        for j in range(centroid.shape[0]):
            distances[j] = np.sum((data[i] - centroid[j])**2)
        assignments[i] = int(np.argmin(distances))
    return assignments

def update_centroids(data, assignments, old_centroids=None):
    # TODO find new centroids based on the assignments
    # with old_centroids given, k is taken from it and a cluster that lost all its points keeps its old
    # centroid (without them an empty cluster would move to the origin, or vanish if it had the highest index)
    k = np.max(assignments)+1 if old_centroids is None else old_centroids.shape[0]
    centroids = np.zeros((k, data.shape[1])) if old_centroids is None else old_centroids.astype(float)
    counts = np.zeros(k)
    sums = np.zeros((k, data.shape[1]))
    # next centroid is the average of the points in its cluster
    for i in range(data.shape[0]):
        sums[assignments[i]] += data[i]
        counts[assignments[i]] += 1
    for i in range(k):
        if counts[i] != 0:
            centroids[i] = sums[i] / counts[i]
    
    return centroids

def mean_intra_distance(data, assignments, centroids):
    # square root of the sum of squared distances to the assigned centroids (the k-means objective)
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False, verbose=True):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        if verbose:
            print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments, centroids)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         


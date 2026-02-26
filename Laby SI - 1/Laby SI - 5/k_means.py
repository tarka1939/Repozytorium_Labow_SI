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
        # next centroid is the one with the max distance
        centroids[i] = data[np.argmax(distances)]        
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

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    centroids = np.zeros((np.max(assignments)+1, data.shape[1]))
    counts = np.zeros(np.max(assignments)+1)
    # next centroid is the average of the points in its cluster
    for i in range(data.shape[0]):
        centroids[assignments[i]] += data[i]
        counts[assignments[i]] += 1
    for i in range(centroids.shape[0]):
        if counts[i] != 0:
            centroids[i] /= counts[i]
    
    return centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         


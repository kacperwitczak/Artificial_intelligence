import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    centroids = np.random.uniform(np.amin(data, axis=0), np.amax(data, axis=0), size=(k, data.shape[1]))

    return centroids

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement Unsupervised classification kmeans++ initizalization
    centroid_index = np.random.choice(data.shape[0], size=1)
    centroids = np.array(data[centroid_index])
    for _ in range(k-1):
        max_distance = -np.inf
        furthest_centroid = None
        for observation in data:
            distance = 0
            for centroid in centroids:
                distance += np.sqrt(np.sum((observation-centroid)**2))
            if distance > max_distance:
                max_distance = distance
                furthest_centroid = observation

        centroids = np.append(centroids, [furthest_centroid], axis=0)

    return centroids

def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    assignments = []
    for observation in data:
        minimum_distance = np.inf
        centroid_index = None
        for i, centroid in enumerate(centroids):
            distance = np.sqrt(np.sum((observation-centroid)**2))
            if distance < minimum_distance:
                minimum_distance = distance
                centroid_index = i

        assignments.append(centroid_index)

    return assignments

def update_centroids(data, assignments):
    groups = [[] for _ in range(max(assignments) + 1)]
    for i, assignment in enumerate(assignments):
        groups[assignment].append(data[i])

    centroids = []
    for group in groups:
        if group:
            centroid = np.mean(group, axis=0)
            centroids.append(centroid)

    return np.array(centroids)

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus=False):
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

    return assignments, centroids, mean_intra_distance(data, assignments, centroids)


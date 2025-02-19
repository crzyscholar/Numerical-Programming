import numpy as np

def k_means_cluster(k, points, max_iters=10):
    # Randomly initialize k centroids from the points
    centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    
    # Initialize clusters list
    clusters = [[] for _ in range(k)]
    
    # Loop until convergence or max iterations
    for _ in range(max_iters):
        # Clear previous clusters
        clusters = [[] for _ in range(k)]
    
        # Assign each point to the "closest" centroid 
        for point in points:
            distances_to_each_centroid = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_assignment = np.argmin(distances_to_each_centroid)
            clusters[cluster_assignment].append(point)
        
        # Calculate new centroids by taking the mean of all points in a cluster
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i] for i, cluster in enumerate(clusters)])
        
        # Check for convergence (i.e., if centroids don't change anymore)
        if np.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids
    
    return clusters, centroids

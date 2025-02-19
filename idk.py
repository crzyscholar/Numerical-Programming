import numpy as np
import matplotlib.pyplot as plt
import time

# Provided dataset with Process Manipulation (X) and Network Communication (Y)
data = np.array([
    [9, 2],   # M1 (Ransomware)
    [8, 3],   # M2 (Ransomware)
    [7, 1],   # M3 (Ransomware)
    [2, 9],   # M4 (Backdoor)
    [1, 10],  # M5 (Backdoor)
    [3, 8],   # M6 (Backdoor)
    [6, 6],   # M7 (Trojan)
    [5, 5],   # M8 (Trojan)
    [4, 7],   # M9 (Trojan)
    [3, 5],   # M10 (Spyware)
    [2, 6],   # M11 (Spyware)
    [1, 7],   # M12 (Spyware)
    [6, 8],   # M13 (Worm)
    [5, 9],   # M14 (Worm)
    [7, 7],   # M15 (Worm)
])

def k_means_cluster(k, points, max_iters=100):
    # Randomly initialize k centroids from the points
    centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    
    # Loop until convergence or max iterations
    for iteration in range(max_iters):
        # Clear previous clusters
        clusters = [[] for _ in range(k)]
    
        # Assign each point to the "closest" centroid 
        for point in points:
            distances_to_each_centroid = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_assignment = np.argmin(distances_to_each_centroid)
            clusters[cluster_assignment].append(point)
        
        # Plot the current state of clusters and centroids
        plot_clusters(clusters, centroids, iteration)

        # Calculate new centroids by taking the mean of all points in a cluster
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i] for i, cluster in enumerate(clusters)])
        
        # Check for convergence (i.e., if centroids don't change anymore)
        if np.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids
        time.sleep(0.5)  # Delay to make the changes easier to observe
    
    return clusters, centroids

def plot_clusters(clusters, centroids, iteration):
    colors = ['r', 'g', 'b', 'y']  # Colors for the different clusters
    plt.figure(figsize=(8, 6))

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        if cluster.size > 0:
            plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i % len(colors)], label=f'Cluster {i+1}')

    # Plotting centroids
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='x', s=100, label='Centroids')

    plt.title(f'Iteration {iteration+1}: Malware Clustering Based on API Calls')
    plt.xlabel('Process Manipulation (X)')
    plt.ylabel('Network Communication (Y)')
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.5)  # Pause to allow graph to update
    plt.clf()  # Clear the figure to prepare for the next iteration

def main():
    k = 4  # Number of clusters (you can easily change this)
    max_iters = 100  # Maximum iterations for K-means
    clusters, centroids = k_means_cluster(k, data, max_iters)

    # Final plot of the results
    plot_clusters(clusters, centroids, max_iters)

if __name__ == "__main__":
    plt.ion()  # Interactive mode on to allow dynamic updates to the plot
    main()
    plt.ioff()  # Turn off interactive mode when done

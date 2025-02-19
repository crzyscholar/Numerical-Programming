import numpy as np
import matplotlib.pyplot as plt

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
    # [6, 8],   # M13 (Worm)
    # [5, 9],   # M14 (Worm)
    # [7, 7],   # M15 (Worm)
])

def k_means_cluster(k, points, max_iters=100):
    centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    
    clusters = [[] for _ in range(k)]
    
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
    
        for point in points:
            distances_to_each_centroid = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_assignment = np.argmin(distances_to_each_centroid)
            clusters[cluster_assignment].append(point)
        
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i] for i, cluster in enumerate(clusters)])
        

        if np.allclose(new_centroids, centroids):
            break
        
        centroids = new_centroids
    
    return clusters, centroids

def plot_clusters(clusters, centroids):
    colors = ['r', 'g', 'b', 'y']
    plt.figure(figsize=(8, 6))

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i % len(colors)], label=f'Cluster {i+1}')


    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='x', s=100, label='Centroids')

    plt.title('Malware Clustering Based on API Calls (K-Means)')
    plt.xlabel('Process Manipulation (X)')
    plt.ylabel('Network Communication (Y)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    k = 4
    max_iters = 100
    clusters, centroids = k_means_cluster(k, data, max_iters)

    plot_clusters(clusters, centroids)

if __name__ == "__main__":
    main()

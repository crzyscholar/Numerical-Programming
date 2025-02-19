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

def k_medoids_cluster(k, points, max_iters=100):
    medoids_indices = np.random.choice(points.shape[0], k, replace=False)
    medoids = points[medoids_indices]
    
    clusters = [[] for _ in range(k)]
    
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
    
        for point in points:
            distances_to_each_medoid = [np.linalg.norm(point - medoid) for medoid in medoids]
            cluster_assignment = np.argmin(distances_to_each_medoid)
            clusters[cluster_assignment].append(point)
        
        # Update medoids
        new_medoids = []
        for cluster in clusters:
            if cluster:
                distances = np.array([np.sum([np.linalg.norm(p - other) for other in cluster]) for p in cluster])
                new_medoids.append(cluster[np.argmin(distances)])
            else:
                new_medoids.append(medoids[len(new_medoids)])
        
        if np.array_equal(new_medoids, medoids):
            break
        
        medoids = np.array(new_medoids)
    
    return clusters, medoids


k = 4
clusters, medoids = k_medoids_cluster(k, data)


colors = ['r', 'g', 'b', 'y']
plt.figure(figsize=(8, 6))

for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f'Cluster {i+1}')


medoids = np.array(medoids)
plt.scatter(medoids[:, 0], medoids[:, 1], color='k', marker='x', s=100, label='Medoids')

plt.title('Malware Clustering Based on API Calls (K-Medoids)')
plt.xlabel('Process Manipulation (X)')
plt.ylabel('Network Communication (Y)')
plt.legend()
plt.grid(True)
plt.show()

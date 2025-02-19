import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

max_k = 4

# Set up the plotting environment
plt.figure(figsize=(8, 6))

for k in range(1, max_k + 1):
    # Initialize KMeans with the desired number of clusters (k)
    kmeans = KMeans(n_clusters=k, random_state=40, n_init=10)
    
    # Fit the KMeans algorithm to the data
    kmeans.fit(data)
    
    # Get the resulting labels and centroids
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Clear the current figure
    plt.clf()
    
    # Plot data points and color them by cluster assignment
    plt.scatter(
        data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k'
    )
    
    # Plot the cluster centers
    plt.scatter(
        centers[:, 0], centers[:, 1],
        c='red', marker='X', s=200, alpha=0.75, label='Cluster Centers'
    )
    
    # Add plot labels and title
    plt.title(f'K-Means Clustering with k={k}')
    plt.xlabel('Process Manipulation (X)')
    plt.ylabel('Network Communication (Y)')
    plt.legend()
    plt.grid(True)
    
    # Show the plot and wait for button press
    plt.draw()
    plt.waitforbuttonpress()

# Show the final plot
plt.show()

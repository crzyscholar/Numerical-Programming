import numpy as np
import cv2

# CP1-dan

def dbscan(X, eps, min_points):
    X = np.asarray(X)
    n_samples = X.shape[0]
    visited = np.zeros(n_samples, dtype=bool)
    clustered = np.zeros(n_samples, dtype=bool)
    clusters = []
    
    def find_neighbors(point_idx):
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= eps)[0]
    
    # mtavari pointebi
    for point_idx in range(n_samples):
        if visited[point_idx]:
            continue
            
        visited[point_idx] = True
        neighbors = find_neighbors(point_idx)
        
        # daskipe ara mtavari pointebi
        if len(neighbors) < min_points:
            continue
            
        current_cluster = [point_idx]
        clustered[point_idx] = True
        
        # mezoblebis gachekva
        seed_points = list(neighbors) 
        
        while seed_points:
            current_point = seed_points.pop(0)
            
            if not visited[current_point]:
                visited[current_point] = True
                point_neighbors = find_neighbors(current_point)
                
                if len(point_neighbors) >= min_points:
                    seed_points.extend([p for p in point_neighbors if p not in seed_points])
            
            if not clustered[current_point]:
                current_cluster.append(current_point)
                clustered[current_point] = True
        
        clusters.append(X[current_cluster])
    
    # datasets - clusterebi = noise
    noise = X[~clustered]
    
    return clusters, noise

# https://en.wikipedia.org/wiki/Canny_edge_detector#Process
def canny_edge_detection(image, low_threshold=20, high_threshold=40, sigma=1):
    # https://en.wikipedia.org/wiki/Gaussian_blur
    def gaussian_kernel(size, sigma): # https://gist.github.com/FienSoP/49b61a7afdcb992fef0c8ef1b2b3364c
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    def non_maximum_suppression(magnitude, direction):
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        angle = np.rad2deg(direction) % 180
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                # daxrilobis mixedvit mezoblebis shemowmeba
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180): # horizontaluri
                    neighbors = [magnitude[i,j-1], magnitude[i,j+1]]
                elif 22.5 <= angle[i,j] < 67.5: # diagnali
                    neighbors = [magnitude[i-1,j-1], magnitude[i+1,j+1]]
                elif 67.5 <= angle[i,j] < 112.5: # vertikaluri
                    neighbors = [magnitude[i-1,j], magnitude[i+1,j]]
                else: # meore diagonali
                    neighbors = [magnitude[i-1,j+1], magnitude[i+1,j-1]]
                
                if magnitude[i,j] >= max(neighbors):
                    suppressed[i,j] = magnitude[i,j]
                    
        return suppressed
    
    def hysteresis_thresholding(img, low, high):
        strong = (img > high)
        weak = (img > low) & (img <= high)
        output = np.zeros_like(img)
        output[strong] = 1
        
        # low-s da high-s shoris rac moxvda magati mezoblebis gachekva
        height, width = img.shape
        for i in range(1, height-1):
            for j in range(1, width-1):
                if weak[i,j]:
                    if np.any(output[i-1:i+2, j-1:j+2] == 1):
                        output[i,j] = 1
                        
        return output
    
    # Gausiani
    kernel_size = int(6*sigma + 1) if int(6*sigma + 1) % 2 == 1 else int(6*sigma + 2)
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = np.zeros_like(image, dtype=float)
    
    padded = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='edge')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            smoothed[i,j] = np.sum(window * kernel)

    # https://en.wikipedia.org/wiki/Sobel_operator
    # sobeli
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gradient_x = np.zeros_like(smoothed)
    gradient_y = np.zeros_like(smoothed)
    
    padded = np.pad(smoothed, ((1, 1), (1, 1)), mode='edge')
    
    for i in range(smoothed.shape[0]):
        for j in range(smoothed.shape[1]):
            window = padded[i:i+3, j:j+3]
            gradient_x[i,j] = np.sum(window * sobel_x)
            gradient_y[i,j] = np.sum(window * sobel_y)
    
    # https://en.wikipedia.org/wiki/Sobel_operator
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.atan2(gradient_y, gradient_x) # aq romeli jobia ar vici (atan2 tu arctan2)
    
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)
    
    return edges

def get_target_coords(path, low_threshold=30, high_threshold=90, sigma=1.0, eps=30, min_points=5):

    def calculate_center(cluster):
        return np.mean(cluster, axis=0)

    frame = cv2.imread(path)
    #frame = cv2.resize(frame, (200, 200))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = canny_edge_detection(gray, low_threshold, high_threshold, sigma)

    edge_pixels = np.column_stack(np.where(edges > 0))

    clusters, _ = dbscan(edge_pixels, eps, min_points)

    centers = []
    for cluster in clusters:
        center = calculate_center(cluster)
        x = center[1]
        y = frame.shape[0] - center[0]
        centers.append([x, y])

    centers.sort(key=lambda c: c[0])

    return centers
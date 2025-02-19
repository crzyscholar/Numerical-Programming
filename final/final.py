import cv2
import numpy as np

def detect_edges(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    grad_magnitude = (grad_magnitude / grad_magnitude.max() * 255).astype(np.uint8)
    _, edges = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)

    return edges

def calculate_distances(pts1, pts2):
    if len(pts1) == 0 or len(pts2) == 0:
        return []
    result = []
    for pt2 in pts2:
        row = []
        for pt1 in pts1:
            dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
            row.append(dist)
        result.append(row)
    return result

def calculate_mean(lst):
    if len(lst) == 0:
        return 0
    total = 0
    for val in lst:
        total += val
    return total / len(lst)

def apply_threshold(diff_frame, threshold_value):
    thresh_frame = diff_frame.copy()
    for y in range(diff_frame.shape[0]):
        for x in range(diff_frame.shape[1]):
            thresh_frame[y, x] = 255 if diff_frame[y, x] > threshold_value else 0
    return thresh_frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    prev_centroids = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, gray)

        thresh = apply_threshold(diff, 25)

        edges = detect_edges(thresh)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))

        if frame_count > 0 and len(prev_centroids) > 0 and len(centroids) > 0:
            distances = calculate_distances(prev_centroids, centroids)
            speeds = [min(row) for row in distances] if distances else []
            print(f"Frame {frame_count}: Moving objects = {len(centroids)}, Speeds = {speeds}")

        prev_centroids = centroids
        prev_gray = gray
        frame_count += 1

    cap.release()

process_video("bad.mp4")


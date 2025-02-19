import cv2
import numpy as np

def calculate_velocity(positions, fps):
    velocities = []
    for i in range(1, len(positions) - 1):
        vx = (positions[i + 1][0] - positions[i - 1][0]) / (2 / fps)
        vy = (positions[i + 1][1] - positions[i - 1][1]) / (2 / fps)
        velocities.append((vx, vy))
    return velocities

def estimate_parameters(velocities, fps):
    k_m_estimates = []
    g_estimates = []
    for i in range(1, len(velocities) - 1):
        ax = (velocities[i + 1][0] - velocities[i - 1][0]) / (2 / fps)
        ay = (velocities[i + 1][1] - velocities[i - 1][1]) / (2 / fps)
        vx, vy = velocities[i]
        k_m = - (ax / (vx * np.sqrt(vx**2 + vy**2))) if vx != 0 else 0
        g = ((- k_m) * vy * np.sqrt(vx**2 + vy**2)) - ay
        k_m_estimates.append(k_m)
        g_estimates.append(g)

    return np.mean(k_m_estimates, dtype=np.float32), np.mean(g_estimates, dtype=np.float32)

def get_velocities_from_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("no frame")
            break

        hg = frame.shape[0]
        wd = frame.shape[1]

        frame = cv2.resize(frame, (int(wd / (hg/200)), 200))
        hg = frame.shape[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (3, 3), 2)

        cv2.imshow("bl", blurred)

        edges = cv2.Canny(blurred, 30, 255)

        cv2.imshow("edges", edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center = (x + w // 2, hg - (y + h // 2))
            positions.append(center)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    velocities = calculate_velocity(positions, fps)

    k_m, g = estimate_parameters(velocities, fps)

    cap.release()

    return k_m, g, velocities, positions

def within_tolerance(x1, y1, x2, y2, tolerance = 0.01):
    return abs(x1 - x2) <= tolerance and abs(y1 - y2) <= tolerance
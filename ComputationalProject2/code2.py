import cv2
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def ball_motion(t, state, g, k, m):
    x, y, vx, vy = state
    v = np.sqrt(vx**2 + vy**2)
    dxdt = vx
    dydt = vy
    dvxdt = -(k/m) * vx * v
    dvydt = -g - (k/m) * vy * v
    return [dxdt, dydt, dvxdt, dvydt]

def track_ball(video_path):
    cap = cv2.VideoCapture(video_path)
    positions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # hsv worked better
        
        lower_color = np.array([30, 50, 50])  
        upper_color = np.array([60, 255, 255])
        
        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour and use it as the ball position
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                positions.append((cX, cY))
    
    cap.release()
    return np.array(positions)

def error_function(params, t, observed_positions, g):
    k, m, vx0, vy0 = params
    
    initial_state = [0.0, 0.0, vx0, vy0]
    
    sol = solve_ivp(
        ball_motion,
        [0, t[-1]],
        initial_state,
        args=(g, k, m),
        t_eval=t,
        method='RK45'  
        )
    
    simulated_positions = np.vstack((sol.y[0], sol.y[1])).T
    
    error = np.linalg.norm(observed_positions - simulated_positions)
    return error

def analyze_video(video_path):
    g = 9.8      
    positions = track_ball(video_path)
    
    if len(positions) < 2:
        raise ValueError("Not enough positions tracked from the video.")
    
    times = np.linspace(0, len(positions) / 30.0, len(positions)) # albat 30 fps idk 
    
    observed_positions = positions[:, :2]  
    
    # Initial guesses for k, m, vx0, vy0 
    initial_guess = [0.00001, 0.5, 10.0, 10.0]
    
    result = minimize(
        error_function,
        initial_guess,
        args=(times, observed_positions, g),
        bounds=[(0.01, 1), (0.1, 5), (-50, 50), (-50, 50)]
    )
    
    k_estimated, m_estimated, vx0_estimated, vy0_estimated = result.x
    
    return k_estimated, m_estimated, vx0_estimated, vy0_estimated

video_path = "video2.mp4"  
k_estimated, m_estimated, vx0_estimated, vy0_estimated = analyze_video(video_path)
# print(f"Estimated Drag Coefficient: {k_estimated}")
# print(f"Estimated Mass: {m_estimated}")
print(f"Estimated Initial Velocity X: {vx0_estimated}")
print(f"Estimated Initial Velocity Y: {vy0_estimated}")


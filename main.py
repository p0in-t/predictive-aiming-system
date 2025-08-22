import random

import cv2
import os, shutil
import numpy as np
import sys, pygame
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from utils import euclidean_norm

M = 0.5
G = 9.81
K = 0.01
PIXELS_TO_METERS = 0.01
WIDTH, HEIGHT = 0, 0

def pixels_to_meters(pixels):
    return pixels * PIXELS_TO_METERS

def meters_to_pixels(meters):
    return meters / PIXELS_TO_METERS

def ball_motion_ode(vx, vy, m, g, k):
    dx = vx
    dy = vy
    dvx = (-k / m) * vx * np.sqrt(vx ** 2 + vy ** 2)
    dvy = -(-g - (k / m) * vy * np.sqrt(vx ** 2 + vy ** 2))
    return dx, dy, dvx, dvy

def clear_dir(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def interpolate_edges(contours, spacing=1):
    points = []

    for i in range(len(contours) - 1):
        pt1 = contours[i][0]
        pt2 = contours[i + 1][0]

        points.append(pt1)

        dist = euclidean_norm(pt1, pt2)
        num_points = max(2, int(dist / spacing))

        for j in range(num_points):
            offset = j / (num_points - 1)
            point = pt1 + offset * (pt2 - pt1)
            points.append(point.astype(int))

        points.append(pt2)

    return np.array(points)

def get_frame(path):
    global WIDTH, HEIGHT

    img = cv2.imread(path)

    WIDTH = img.shape[0]
    HEIGHT = img.shape[1]

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    copy = np.array(img)

    blurred = cv2.GaussianBlur(copy, (11, 11), 0)

    # sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    # sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    #
    # magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # magnitude = np.uint8(magnitude * 255 / np.max(magnitude))

    canny = cv2.Canny(image=cv2.convertScaleAbs(blurred), threshold1=140, threshold2=220)

    # cleaned = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel_small, iterations=8)
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_mid, iterations=8)
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=8)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    cv2.drawContours(copy, filtered, -1, (0, 255, 0), 2)

    cv2.imshow('Canny', canny)
    cv2.imshow('Contours', copy)

    cv2.waitKey(100000)

    return img, contours

def center_of_mass(cluster):
    return np.mean(cluster, axis=0).astype(int)

def recognize_clusters(points, labels):
    centers = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue

        cluster_points = points[labels == label]
        centers.append(center_of_mass(cluster_points))

    return centers

def get_positions(path):
    frame, contours = get_frame(path)

    dbscan = DBSCAN(eps=3, min_samples=4)
    positions = []
    radius_arr = []
    wait_time = 5000

    all_points = []
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        rect = cv2.minAreaRect(contour)
        width = min(rect[1])
        height = max(rect[1])
        aspect_ratio = width / height if height > 0 else 0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        MIN_CIRCULARITY = 0.4
        MIN_ASPECT_RATIO = 0.4
        MIN_CONVEXITY = 0.7

        is_sphere = (circularity > MIN_CIRCULARITY and aspect_ratio > MIN_ASPECT_RATIO and convexity > MIN_CONVEXITY)

        if is_sphere:
            points = interpolate_edges(contour, spacing=1)
            if len(points) > 0:
                all_points.extend(points)
                valid_contours.append(contour)

    points = np.array(all_points)

    if points.size == 0:
        cv2.putText(frame, f"Number of objects: {0}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 0, 0), 2)
        cv2.imshow('Object detection', frame)
        cv2.waitKey(wait_time)
        return None

    labels = dbscan.fit_predict(points)
    centers = recognize_clusters(points, labels)
    unique_labels = set(labels)
    colors = [(255, 0, 0), (0, 255, 0)]

    cv2.putText(frame, f"Number of objects: {len(unique_labels)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
               (255, 0, 0), 2)

    for label in unique_labels:
        if label == -1:
            cv2.imshow('Object detection', frame)
            cv2.waitKey(wait_time)
            continue

        color = colors[label % len(colors)]
        cluster_points = points[labels == label]

        for point in cluster_points:
            cv2.circle(frame, tuple(point), 2, color, -1)

    for j, center in enumerate(centers):
        positions.append(center)
        area = cv2.contourArea(valid_contours[j])
        radius = int(np.sqrt(area / np.pi))
        radius_arr.append(radius)
        cv2.circle(frame, tuple(center), radius, (0, 255, 255), 2)

        cv2.circle(frame, tuple(center), 5, (0, 255, 255), -1)

    cv2.imshow('Object detection', frame)
    cv2.waitKey(wait_time)

    cv2.destroyAllWindows()

    return np.array(positions), np.array(radius_arr)

def rk4(state, delta_time, m, g, k):
    vx, vy = state[2], state[3]
    k1x, k1y, k1vx, k1vy = ball_motion_ode(vx, vy, m, g, k)
    k2x, k2y, k2vx, k2vy = ball_motion_ode(vx + k1vx * delta_time / 2, vy + k1vy * delta_time / 2, m, g, k)
    k3x, k3y, k3vx, k3vy = ball_motion_ode(vx + k2vx * delta_time / 2, vy + k2vy * delta_time / 2, m, g, k)
    k4x, k4y, k4vx, k4vy = ball_motion_ode(vx + k3vx * delta_time, vy + k3vy * delta_time, m, g, k)

    dx = (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    dy = (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    dvx = (k1vx + 2 * k2vx + 2 * k3vx + k4vx) / 6
    dvy = (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6

    return dx * delta_time, dy * delta_time, dvx * delta_time, dvy * delta_time

def forward_euler(state, delta_time, m, g, k):
    vx, vy = state[2], state[3]
    dx, dy, dvx, dvy = ball_motion_ode(vx, vy, m, g, k)

    return dx * delta_time, dy * delta_time, dvx * delta_time, dvy * delta_time

def trapezoidal(state, delta_time, m, g, k, max_iterations=200, tolerance=1e-6):
    x, y, vx, vy = state
    curr_dx, curr_dy, curr_dvx, curr_dvy = ball_motion_ode(vx, vy, m, g, k)

    next_x = x + curr_dx * delta_time
    next_y = y + curr_dy * delta_time
    next_vx = vx + curr_dvx * delta_time
    next_vy = vy + curr_dvy * delta_time

    for _ in range(max_iterations):
        prev_x, prev_y = next_x, next_y
        prev_vx, prev_vy = next_vx, next_vy

        next_dx, next_dy, next_dvx, next_dvy = ball_motion_ode(next_vx, next_vy, m, g, k)

        next_x = x + 0.5 * delta_time * (curr_dx + next_dx)
        next_y = y + 0.5 * delta_time * (curr_dy + next_dy)
        next_vx = vx + 0.5 * delta_time * (curr_dvx + next_dvx)
        next_vy = vy + 0.5 * delta_time * (curr_dvy + next_dvy)

        x_diff = abs(next_x - prev_x)
        y_diff = abs(next_y - prev_y)
        vx_diff = abs(next_vx - prev_vx)
        vy_diff = abs(next_vy - prev_vy)

        if x_diff < tolerance and y_diff < tolerance and vx_diff < tolerance and vy_diff < tolerance:
            break

    dx = next_x - x
    dy = next_y - y
    dvx = next_vx - vx
    dvy = next_vy - vy

    return dx, dy, dvx, dvy

def normalize_pos(pos, r, w, h):
    if pos[0] <= r:
        pos[0] = r
    elif pos[0] > w-r:
        pos[0] = w-r

    if pos[1] <= r:
        pass
        # pos[1] = r
    elif pos[1] > h-r:
        pos[1] = h-r

def residual(simulated_pos, target_pos):
    simulated_x, simulated_y = simulated_pos

    simulated_end = np.array([simulated_x[-1], simulated_y[-1]])

    residuals = np.zeros(2)
    residuals[0] = simulated_end[0] - target_pos[0]
    residuals[1] = simulated_end[1] - target_pos[1]

    return residuals


def estimate_max_iterations(window_size, velocity_bounds, delta_time):
    width_m, height_m = window_size

    max_vx = max(abs(velocity_bounds[0][0]), abs(velocity_bounds[0][1]))
    max_vy = max(abs(velocity_bounds[1][0]), abs(velocity_bounds[1][1]))

    diagonal_distance = np.sqrt(width_m ** 2 + height_m ** 2)
    max_velocity = np.sqrt(max_vx ** 2 + max_vy ** 2)

    min_time = (diagonal_distance / max_velocity) * 1.5

    max_iters = int(min_time / delta_time)

    return max_iters

def simulate_trajectory(initial_state, params, delta_time, window_size, velocity_bounds):
    M, G, K = params

    x = [initial_state[0]]
    y = [initial_state[1]]
    vx = initial_state[2]
    vy = initial_state[3]

    max_iters = estimate_max_iterations(window_size, velocity_bounds, delta_time)

    for i in range(0, max_iters):
        dx, dy, dvx, dvy = forward_euler([x[-1], y[-1], vx, vy], delta_time, M, G, K)

        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        vx += dvx
        vy += dvy

    return np.array(x), np.array(y)

def numerical_jacobian(func, x, epsilon=1e-8):
    f = np.array(func(x))

    n = len(x)
    m = len(f)
    J = np.zeros((m, n))

    for i in range(n):
        h = np.zeros(n)
        h[i] = epsilon

        f_plus = func(x + h)
        f_minus = func(x - h)
        J[:, i] = (f_plus - f_minus) / (2 * epsilon)

    return f, J

def newton_shooting_method(initial_velocities, fixed_params, start_pos, target_pos, delta_time, window_size, max_iters=10, tol=1e-8):
    params = initial_velocities.copy()
    x0, y0 = start_pos
    m, g, k = fixed_params

    bounds = np.array([
        [-20.0, 20.0],
        [-20.0, 20.0]
    ])

    for iter in range(max_iters):
        def res(p):
            vx0_test, vy0_test = p
            initial_state = np.array([x0, y0, vx0_test, vy0_test])
            simulated_x, simulated_y = simulate_trajectory(initial_state,[m, g, k], delta_time, window_size, bounds)
            return residual((simulated_x, simulated_y), target_pos)

        error, J = numerical_jacobian(res, params)

        lambda_reg = 1e-6
        delta = np.linalg.solve(J.T @ J + lambda_reg * np.eye(len(params)), -J.T @ error)

        new_params = params + delta
        new_params = np.clip(new_params, bounds[:, 0], bounds[:, 1])

        residual_norm = euclidean_norm(error, np.zeros(error.shape))

        if residual_norm < tol:
            break

        params = new_params

    return params

def analyze_frame(positions, local_pos, window_size, velocity_bounds):
    global M, G, K

    positions = np.array([[pixels_to_meters(x[0]), pixels_to_meters(x[1])] for x in positions])

    velocities = []

    initial_velocities = np.array([
        0.0,
        0.0
    ])

    delta_time = 0.01

    for idx, pos in enumerate(positions):
        result = newton_shooting_method(initial_velocities, [M, G, K], local_pos, pos, delta_time, window_size)
        velocities.append(result)
        print(f"Optimal parameters for position {idx}: {result}")

    for idx, pos in enumerate(positions):
        initial_state = np.array([
            local_pos[0],
            local_pos[1],
            velocities[idx][0],
            velocities[idx][1]
        ])

        simulated_x, simulated_y = simulate_trajectory(initial_state, [M, G, K], delta_time, window_size, velocity_bounds)

        mse_x = np.mean((simulated_x[-1] - positions[idx, 0]) ** 2)
        mse_y = np.mean((simulated_y[-1] - positions[idx, 1]) ** 2)
        total_mse = mse_x + mse_y

        print(f"\nError metrics:")
        print(f"MSE X: {mse_x:.4f}")
        print(f"MSE Y: {mse_y:.4f}")
        print(f"Total MSE: {total_mse:.4f}\n-----------------------------\n")

    return positions, velocities

def circle_collision(pos1, pos2, vel1, vel2, radius, damping):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    distance = np.sqrt(dx * dx + dy * dy)

    if distance < radius * 2:
        overlap = (radius * 2 - distance) / 2
        nx = dx / distance
        ny = dy / distance

        pos1[0] += nx * overlap
        pos1[1] += ny * overlap
        pos2[0] -= nx * overlap
        pos2[1] -= ny * overlap

        temp_vel = vel1.copy()
        vel1_new = vel2 * damping
        vel2_new = temp_vel * damping

        return vel1_new, vel2_new

    return vel1, vel2


def ball_sim(positions, local_circle_radius, radius_arr, params, local_pos):
    global WIDTH, HEIGHT
    size = width, height = HEIGHT, WIDTH

    pygame.init()
    pygame.font.init()
    my_font = pygame.font.SysFont('Tahoma', 30, True)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()

    p_positions = np.array([[meters_to_pixels(x[0]), meters_to_pixels(x[1])] for x in positions])
    p_params = np.array([[meters_to_pixels(x[0]), meters_to_pixels(x[1])] for x in params])
    p_local_pos = [meters_to_pixels(local_pos[0]), meters_to_pixels(local_pos[1])]
    p_radius_arr = radius_arr

    circle_pos = np.array([p_local_pos], dtype=np.float64)
    current_target = 0
    circle_speed = np.array([p_params[current_target]], dtype=np.float64)

    trajectory_surface = pygame.Surface(size, pygame.SRCALPHA)
    trajectory_points = []

    ball_color = (233, 22, 27)
    target_color = (0, 255, 0)
    hit_color = (128, 128, 128)
    trajectory_color = (233, 22, 27, 100)
    black = 255, 255, 255
    damping = 0.7

    targets_hit = [False] * len(p_positions)

    debug = False
    reset = False
    last_hit_time = 0
    hit_cooldown = 0.5

    WAITING = "waiting"
    MOVING = "moving"
    current_state = WAITING
    wait_start_time = pygame.time.get_ticks() / 1000
    wait_duration = 0.5

    while True:
        delta_time = clock.tick(120) / 1000
        fps = clock.get_fps()
        current_time = pygame.time.get_ticks() / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    debug = not debug
                if event.key == pygame.K_r:
                    reset = True

        if reset:
            circle_pos = np.array([p_local_pos], dtype=np.float64)
            current_target = 0
            circle_speed = np.array([p_params[current_target]], dtype=np.float64)
            targets_hit = [False] * len(p_positions)
            last_hit_time = 0
            current_state = WAITING
            wait_start_time = current_time
            reset = False
            trajectory_surface.fill((0, 0, 0, 0))
            trajectory_points = []
            continue

        if current_state == WAITING:
            circle_pos = np.array([p_local_pos], dtype=np.float64)
            circle_speed = np.array([p_params[current_target]], dtype=np.float64)
            trajectory_points = []

            if current_time - wait_start_time >= wait_duration:
                current_state = MOVING

        elif current_state == MOVING:
            tx, ty = np.array([pixels_to_meters(circle_pos[0][0]), pixels_to_meters(circle_pos[0][1])])
            tvx, tvy = np.array([pixels_to_meters(circle_speed[0][0]), pixels_to_meters(circle_speed[0][1])])

            dx, dy, dvx, dvy = forward_euler([tx, ty, tvx, tvy], delta_time, M, G, K)

            prev_pos = circle_pos[0].copy()

            circle_pos[0][0] += meters_to_pixels(dx)
            circle_pos[0][1] += meters_to_pixels(dy)
            circle_speed[0][0] += meters_to_pixels(dvx)
            circle_speed[0][1] += meters_to_pixels(dvy)

            trajectory_points.append((int(prev_pos[0]), int(prev_pos[1])))

            if len(trajectory_points) > 1:
                pygame.draw.line(trajectory_surface, trajectory_color,
                                 trajectory_points[-2], trajectory_points[-1], 2)

            circle_bottom = circle_pos[0][1] + local_circle_radius
            circle_left = circle_pos[0][0] - local_circle_radius
            circle_right = circle_pos[0][0] + local_circle_radius
            circle_top = circle_pos[0][1] - local_circle_radius

            if circle_bottom >= height:
                circle_speed[0][1] = -circle_speed[0][1] * damping
                circle_pos[0][1] = height - local_circle_radius

            if circle_left < 0 or circle_right > width:
                circle_speed[0][0] = -circle_speed[0][0] * damping

            if circle_top > height:
                circle_speed[0][1] = -circle_speed[0][1] * damping

            normalize_pos(circle_pos[0], local_circle_radius, width, height)

            if current_target < len(p_positions) and not targets_hit[current_target]:
                target_pos = p_positions[current_target]
                dx = circle_pos[0][0] - target_pos[0]
                dy = circle_pos[0][1] - target_pos[1]
                distance = np.sqrt(dx * dx + dy * dy)

                if distance < (local_circle_radius + p_radius_arr[current_target]) and (
                        current_time - last_hit_time) > hit_cooldown:
                    targets_hit[current_target] = True
                    current_target += 1
                    last_hit_time = current_time

                    if current_target < len(p_positions):
                        current_state = WAITING
                        wait_start_time = current_time

        screen.fill(black)

        screen.blit(trajectory_surface, (0, 0))

        for i, pos in enumerate(p_positions):
            color = hit_color if targets_hit[i] else target_color
            pygame.draw.circle(screen, color, pos, p_radius_arr[i])

        pygame.draw.circle(screen, ball_color, circle_pos[0], local_circle_radius)

        if debug:
            fps_surface = my_font.render(f'FPS: {fps:.1f}', True, (0, 0, 0))
            targets_surface = my_font.render(f'Target: {current_target}/{len(positions)}', True, (0, 0, 0))
            state_surface = my_font.render(f'State: {current_state}', True, (0, 0, 0))

            if current_state == WAITING:
                time_left = wait_duration - (current_time - wait_start_time)
                time_surface = my_font.render(f'Wait: {time_left:.1f}s', True, (0, 0, 0))
                screen.blit(time_surface, (0, 90))

            screen.blit(fps_surface, (0, 0))
            screen.blit(targets_surface, (0, 30))
            screen.blit(state_surface, (0, 60))

        pygame.display.flip()

def run_sim(path):
    global WIDTH, HEIGHT

    velocity_bounds = np.array([
        [-20.0, 20.0],
        [-20.0, 20.0]
    ])

    frame_pos, radius_arr = get_positions(path)
    local_radius = 15
    window_size = (pixels_to_meters(WIDTH), pixels_to_meters(HEIGHT))

    while True:
        local_pos = [pixels_to_meters(random.randint(30 + local_radius, HEIGHT - 30 - local_radius)),
                     pixels_to_meters(random.randint(30 + local_radius, WIDTH - 30 - local_radius))]
        positions, params = analyze_frame(frame_pos, local_pos, window_size, velocity_bounds)
        ball_sim(positions, local_radius, radius_arr, params, local_pos)

def main():
    run_sim("./data/1.jpg")

if __name__ == '__main__':
    main()
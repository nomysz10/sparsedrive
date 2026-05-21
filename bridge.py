"""Most BeamNG ↔ SparseDrive.

Integruje model SparseDrive z symulatorem BeamNG.tech:
- SparseDriveBridge: ładuje model, uruchamia inferencję
- InputBuilder: konwertuje obrazy BeamNG do formatu nuScenes
- ControlExtractor: wyciąga steering/throttle/brake z trajektorii (Pure Pursuit)
- FallbackPerception: awaryjna percepcja gdy model niedostępny
"""
import time
import math
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import (
    INPUT_SHAPE, IMG_NORM_MEAN, IMG_NORM_STD, NUM_CAMS, QUEUE_LENGTH,
    CAM_INTRINSIC, CAM_IMAGE_SIZE, CAM_CONFIGS,
    LOOKAHEAD_DIST, LANE_WIDTH, LANE_CHANGE_DURATION,
    STEER_KP, STEER_KI, STEER_KD,
    SPEED_KP, SPEED_KI, SPEED_KD,
    TARGET_SPEED_DEFAULT, TARGET_SPEED_MAX, TARGET_SPEED_MIN,
    EMERGENCY_BRAKE_DIST, EMERGENCY_BRAKE_FORCE,
    CMD_GO_STRAIGHT, CMD_TURN_LEFT, CMD_TURN_RIGHT,
    CMD_TO_ONEHOT,
)


# ==============================================================================
# FALLBACK PERCEPTION — gdy model SparseDrive niedostępny
# ==============================================================================

class FallbackPerception:
    """Klasyczna percepcja z 6 kamer jako fallback.

    Wykrywa: pasy ruchu (Canny+Hough), pojazdy (kontury+kolor), wolną przestrzeń.
    Używane tylko gdy model SparseDrive nie może być załadowany.
    """

    def __init__(self):
        self._detected_vehicles = []
        self._lane_center = None
        self._road_edges = None

    def process(self, images: dict, vehicle_yaw: float):
        """Przetwarza obrazy z 6 kamer i zwraca wyniki w formacie zbliżonym do SparseDrive.

        Returns:
            dict z kluczami: boxes_3d, vectors, final_planning, lane_available
        """
        front = images.get('F')
        if front is None:
            return self._empty_result()

        h, w = front.shape[:2]

        # 1. Wykrywanie pasów z przedniej kamery
        lane_center, left_xs, right_xs = self._detect_lanes(front)

        # 2. Wykrywanie pojazdów na wszystkich kamerach
        vehicles = []
        for cam_key, img in images.items():
            if img is not None:
                vhs = self._detect_vehicles_in_image(img, cam_key)
                vehicles.extend(vhs)

        # 3. Sprawdzenie wolnych pasów
        lane_left_free = self._check_lane_free(vehicles, 'left')
        lane_right_free = self._check_lane_free(vehicles, 'right')

        # 4. Generowanie trajektorii (prosta lub skręt)
        planning_traj = self._generate_trajectory(lane_center, w, h)

        return {
            'boxes_3d': np.array(vehicles) if vehicles else np.zeros((0, 7)),
            'vectors': None,
            'final_planning': np.array(planning_traj),
            'lane_available': {'left': lane_left_free, 'right': lane_right_free},
            'obstacle_ahead': self._check_obstacle_ahead(vehicles),
            'obstacle_distance': self._get_obstacle_distance(vehicles),
        }

    def _detect_lanes(self, img_bgr):
        """Ulepszone wykrywanie pasów (Canny + Hough + polyfit)."""
        h, w = img_bgr.shape[:2]
        roi_top = int(h * 0.55)
        roi = img_bgr[roi_top:h, 0:w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=30, maxLineGap=50)

        left_pts = []
        right_pts = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3:
                    continue

                if slope < -0.3:  # Lewy pas
                    left_pts.extend([(x1, y1), (x2, y2)])
                elif slope > 0.3:  # Prawy pas
                    right_pts.extend([(x1, y1), (x2, y2)])

        roi_h = h - roi_top
        left_x = None
        right_x = None

        if len(left_pts) >= 4:
            left_pts_arr = np.array(left_pts)
            coeffs = np.polyfit(left_pts_arr[:, 1], left_pts_arr[:, 0], 2)
            left_x = np.polyval(coeffs, roi_h // 2)

        if len(right_pts) >= 4:
            right_pts_arr = np.array(right_pts)
            coeffs = np.polyfit(right_pts_arr[:, 1], right_pts_arr[:, 0], 2)
            right_x = np.polyval(coeffs, roi_h // 2)

        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2.0
        elif left_x is not None:
            lane_center = left_x + 60
        elif right_x is not None:
            lane_center = right_x - 60
        else:
            lane_center = w / 2.0

        return lane_center, left_x, right_x

    def _detect_vehicles_in_image(self, img_bgr, cam_key):
        """Wykrywa pojazdy na pojedynczej kamerze (kontury + kolor)."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 40, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vehicles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400:  # za małe
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / (bw + 0.001)
            if 0.5 < aspect < 2.5:  # proporcje auta
                # Przybliżona pozycja 3D względem ego
                rel_x = (x - w / 2) / w * 20  # ±10m
                rel_y = -(y / h) * 30 - 5     # odległość
                rel_w = bw / w * 6
                rel_h = bh / h * 6
                vehicles.append([rel_x, rel_y, 0, rel_w, rel_h, 1.5, 0])

        return vehicles

    def _check_lane_free(self, vehicles, direction):
        """Sprawdza czy sąsiedni pas jest wolny."""
        if not vehicles:
            return True
        offset = -LANE_WIDTH if direction == 'left' else LANE_WIDTH
        for v in vehicles:
            vx = v[0]
            if abs(vx - offset) < LANE_WIDTH and v[1] > -10:
                return False
        return True

    def _check_obstacle_ahead(self, vehicles):
        """Sprawdza czy przed nami jest przeszkoda."""
        for v in vehicles:
            if abs(v[0]) < 2.0 and -15 < v[1] < 0:
                return True
        return False

    def _get_obstacle_distance(self, vehicles):
        """Zwraca odległość do najbliższej przeszkody."""
        min_dist = 100.0
        for v in vehicles:
            if abs(v[0]) < 2.0 and v[1] < 0:
                dist = abs(v[1])
                if dist < min_dist:
                    min_dist = dist
        return min_dist if min_dist < 100 else None

    def _generate_trajectory(self, lane_center, img_w, img_h):
        """Generuje prostą trajektorię 6 punktów (3s horyzont)."""
        error = (lane_center - img_w / 2) / (img_w / 2)  # -1..1
        points = []
        for t in range(6):
            x = error * t * 0.5
            y = t * 3.0  # 3m na krok = 18m zasięg
            points.append([x, y])
        return points

    def _empty_result(self):
        return {
            'boxes_3d': np.zeros((0, 7)),
            'vectors': None,
            'final_planning': np.array([[0, 3], [0, 6], [0, 9], [0, 12], [0, 15], [0, 18]]),
            'lane_available': {'left': True, 'right': True},
            'obstacle_ahead': False,
            'obstacle_distance': None,
        }


# ==============================================================================
# INPUT BUILDER — konwersja obrazów BeamNG do formatu modelu
# ==============================================================================

class InputBuilder:
    """Buduje słownik wejściowy w formacie nuScenes dla modelu SparseDrive."""

    def __init__(self):
        self.input_shape = INPUT_SHAPE  # (704, 256) - (w, h)
        self.mean = torch.tensor(IMG_NORM_MEAN, dtype=torch.float32).view(1, 1, 1, 3)
        self.std = torch.tensor(IMG_NORM_STD, dtype=torch.float32).view(1, 1, 1, 3)

    def build(self, images_6cam: dict, vehicle_state: dict, command: int):
        """Konwertuje obrazy BeamNG na tensor wejściowy modelu.

        Args:
            images_6cam: dict z obrazami OpenCV (BGR, uint8)
            vehicle_state: dict ze stanem pojazdu z BeamNG
            command: int - komenda nawigacyjna

        Returns:
            dict z kluczami: img, projection_mat, image_wh, ego_status, gt_ego_fut_cmd
        """
        batch_imgs = []
        for cam_key in ['FL', 'F', 'FR', 'BL', 'B', 'BR']:
            img = images_6cam.get(cam_key)
            if img is None:
                img = np.zeros((900, 1600, 3), dtype=np.uint8)

            # BGR → RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize do input_shape
            img_resized = cv2.resize(img_rgb, self.input_shape)  # (256, 704, 3)
            # Normalizacja
            img_tensor = torch.from_numpy(img_resized).float()
            img_tensor = (img_tensor - self.mean) / self.std
            # HWC → CHW
            img_tensor = img_tensor.permute(2, 0, 1)
            batch_imgs.append(img_tensor)

        # Stack: (6, 3, 256, 704) → (1, 6, 3, 256, 704)
        img_batch = torch.stack(batch_imgs).unsqueeze(0)

        # Macierze projekcji (syntetyczne)
        projection_mat = self._build_projection_matrices()

        # Ego status
        ego_status = self._extract_ego_status(vehicle_state)

        # Komenda jako one-hot
        cmd_onehot = CMD_TO_ONEHOT.get(command, [0, 0, 1])
        gt_ego_fut_cmd = torch.tensor(cmd_onehot, dtype=torch.float32).unsqueeze(0)

        return {
            'img': img_batch,
            'projection_mat': projection_mat,
            'image_wh': (self.input_shape[0], self.input_shape[1]),
            'ego_status': ego_status,
            'gt_ego_fut_cmd': gt_ego_fut_cmd,
        }

    def _build_projection_matrices(self):
        """Buduje syntetyczne macierze projekcji dla 6 kamer.

        Format nuScenes: (bs, num_cams, 3, 4) — intrinsics @ extrinsics
        """
        proj_mats = []
        for cam_key in ['FL', 'F', 'FR', 'BL', 'B', 'BR']:
            cfg = CAM_CONFIGS[cam_key]
            pos = np.array(cfg['pos'])
            direction = np.array(cfg['dir'])

            # Extrinsics: world → camera
            yaw = math.atan2(direction[0], -direction[1])
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            R = np.array([
                [cos_y, sin_y, 0],
                [-sin_y, cos_y, 0],
                [0, 0, 1],
            ])
            t = -R @ pos.reshape(3, 1)
            extrinsic = np.hstack([R, t])

            # Intrinsics
            K = np.array([
                [800, 0, 800],
                [0, 800, 450],
                [0, 0, 1],
            ])

            # Resize intrinsics do input_shape
            scale_x = self.input_shape[0] / 1600
            scale_y = self.input_shape[1] / 900
            K_resized = K.copy()
            K_resized[0] *= scale_x
            K_resized[1] *= scale_y

            proj_mat = K_resized @ extrinsic
            proj_mats.append(proj_mat)

        return torch.tensor(np.stack(proj_mats), dtype=torch.float32).unsqueeze(0)

    def _extract_ego_status(self, vehicle_state):
        """Wyciąga status ego z vehicle.state (prędkość, przyspieszenie, yaw rate)."""
        vel = vehicle_state.get('vel', (0, 0, 0))
        speed = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        acc = vehicle_state.get('acc', (0, 0, 0))
        yaw_rate = vehicle_state.get('yaw_rate', 0)

        return torch.tensor([[speed, acc[0], acc[1], acc[2], yaw_rate]],
                            dtype=torch.float32)


# ==============================================================================
# CONTROL EXTRACTOR — trajektoria → steering, throttle, brake
# ==============================================================================

class ControlExtractor:
    """Wyciąga sterowanie z trajektorii planowanej przez model (Pure Pursuit).

    Używa final_planning z modelu SparseDrive (6 punktów XY, horyzont 3s).
    Modyfikuje trajektorię dla zmiany pasa (dodaje lateralny offset).
    """

    def __init__(self):
        self._steer_integral = 0.0
        self._steer_prev_error = 0.0
        self._speed_integral = 0.0
        self._speed_prev_error = 0.0
        self._last_time = time.time()

    def extract(self, results: dict, current_speed_ms: float,
                cmd_system=None) -> tuple:
        """Wyciąga steering, throttle, brake z wyników modelu.

        Args:
            results: dict z kluczem 'final_planning' (N,2) lub 'planning'
            current_speed_ms: aktualna prędkość w m/s
            cmd_system: CommandSystem (do obsługi zmiany pasa)

        Returns:
            (steering, throttle, brake, trajectory_2d, target_speed_ms)
        """
        dt = time.time() - self._last_time
        self._last_time = time.time()
        dt = max(0.01, min(0.2, dt))

        # 1. Wyciągnij trajektorię
        planning_traj = None
        if results is not None:
            if 'final_planning' in results:
                planning_traj = results['final_planning']
            elif 'planning' in results:
                plan = results['planning']
                if hasattr(plan, 'cpu'):
                    plan = plan.cpu().numpy()
                if plan.ndim >= 3:
                    plan = plan[0][0]
                planning_traj = plan

        if planning_traj is None:
            planning_traj = np.array([[0, 3], [0, 6], [0, 9], [0, 12], [0, 15], [0, 18]])

        if hasattr(planning_traj, 'cpu'):
            planning_traj = planning_traj.cpu().numpy()

        # 2. Zastosuj offset zmiany pasa
        if cmd_system is not None and cmd_system.is_lane_changing:
            offset = cmd_system.lane_change_offset
            plan_modified = planning_traj.copy()
            n_pts = len(plan_modified)
            for i in range(n_pts):
                progress = (i + 1) / n_pts
                plan_modified[i] = plan_modified[i] + np.array([offset * progress, 0])
            planning_traj = plan_modified

        # 3. Pure Pursuit: znajdź punkt lookahead
        target_steer = self._pure_pursuit_steer(planning_traj, LOOKAHEAD_DIST)

        # 4. PID dla skrętu
        steering = self._steer_pid(target_steer, dt)

        # 5. Prędkość docelowa na podstawie krzywizny trajektorii
        curvature = self._estimate_curvature(planning_traj)
        target_speed = TARGET_SPEED_DEFAULT
        if curvature > 0.01:
            target_speed = max(TARGET_SPEED_MIN, TARGET_SPEED_DEFAULT / (1 + curvature * 80))
        target_speed = min(target_speed, TARGET_SPEED_MAX)

        # 6. PID dla prędkości (throttle)
        speed_error = target_speed - current_speed_ms
        throttle = self._speed_pid(speed_error, dt)
        throttle = max(0.0, min(1.0, throttle))

        # 7. Hamowanie awaryjne
        brake = 0.0
        obstacle_dist = None
        if results and 'obstacle_distance' in results:
            obstacle_dist = results.get('obstacle_distance')
        if obstacle_dist and obstacle_dist < EMERGENCY_BRAKE_DIST:
            brake = EMERGENCY_BRAKE_FORCE
            throttle = 0.0

        if cmd_system is not None and cmd_system.emergency_brake:
            brake = EMERGENCY_BRAKE_FORCE
            throttle = 0.0

        # 8. Trajektoria w pikselach dla wizualizacji na przedniej kamerze
        trajectory_2d = self._project_trajectory_to_image(planning_traj)

        return (
            float(np.clip(steering, -1.0, 1.0)),
            float(throttle),
            float(brake),
            trajectory_2d,
            target_speed,
        )

    def _pure_pursuit_steer(self, traj, lookahead):
        """Oblicza kąt skrętu metodą Pure Pursuit."""
        if traj is None or len(traj) == 0:
            return 0.0
        # Znajdź punkt na trajektorii najbliższy lookahead
        best_idx = 0
        best_dist = float('inf')
        for i, pt in enumerate(traj):
            dist = abs(math.hypot(pt[0], pt[1]) - lookahead)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        target = traj[best_idx]
        angle = math.atan2(target[0], abs(target[1]) + 0.1)
        return angle

    def _estimate_curvature(self, traj):
        """Szacuje krzywiznę trajektorii."""
        if len(traj) < 3:
            return 0.0
        # Krzywizna z 3 punktów
        a = traj[0]
        b = traj[len(traj) // 2]
        c = traj[-1]
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        ab = math.hypot(b[0] - a[0], b[1] - a[1])
        bc = math.hypot(c[0] - b[0], c[1] - b[1])
        ca = math.hypot(a[0] - c[0], a[1] - c[1])
        denom = ab * bc * ca
        if denom < 0.001:
            return 0.0
        return 4 * area / denom

    def _steer_pid(self, target, dt):
        """PID dla skrętu."""
        error = target
        self._steer_integral += error * dt
        self._steer_integral = max(-1.0, min(1.0, self._steer_integral))
        derivative = (error - self._steer_prev_error) / dt if dt > 0 else 0
        self._steer_prev_error = error
        return (STEER_KP * error +
                STEER_KI * self._steer_integral +
                STEER_KD * derivative)

    def _speed_pid(self, error, dt):
        """PID dla prędkości."""
        self._speed_integral += error * dt
        self._speed_integral = max(-1.0, min(1.0, self._speed_integral))
        derivative = (error - self._speed_prev_error) / dt if dt > 0 else 0
        self._speed_prev_error = error
        return (SPEED_KP * error +
                SPEED_KI * self._speed_integral +
                SPEED_KD * derivative)

    def _project_trajectory_to_image(self, traj):
        """Rzutuje trajektorię 3D na obraz przedniej kamery (dla wizualizacji).

        Używa uproszczonej projekcji perspektywicznej.
        """
        if traj is None or len(traj) == 0:
            return None
        points_2d = []
        fx, fy = 800, 800
        cx, cy = 800, 450
        for pt in traj:
            x, y = pt[0], abs(pt[1]) + 2.0
            z = -0.5
            px = fx * x / y + cx
            py = fy * z / y + cy
            points_2d.append([px, py])
        return np.array(points_2d, dtype=np.float32)

    def reset(self):
        self._steer_integral = 0.0
        self._steer_prev_error = 0.0
        self._speed_integral = 0.0
        self._speed_prev_error = 0.0


# ==============================================================================
# SPARSEDRIVE BRIDGE — główna klasa ładująca model
# ==============================================================================

class SparseDriveBridge:
    """Ładuje model SparseDrive i udostępnia inferencję.

    Jeśli model nie może być załadowany (brak mmcv, checkpointów, CUDA ops),
    używa FallbackPerception.
    """

    def __init__(self, config_path=None, checkpoint_path=None):
        self.model = None
        self.use_fallback = True
        self.fallback = FallbackPerception()
        self.input_builder = InputBuilder()
        self.control_extractor = ControlExtractor()
        self._temporal_queue = deque(maxlen=QUEUE_LENGTH)
        self._model_loaded = False

        # Próbuj załadować model SparseDrive
        if config_path and checkpoint_path:
            self._try_load_model(config_path, checkpoint_path)

    def _try_load_model(self, config_path, checkpoint_path):
        """Ładuje model SparseDrive z mmcv-full 1.7.1."""
        try:
            import sys
            import os

            sparsedrive_dir = os.path.join(os.path.dirname(__file__), 'SparseDrive-main')
            sys.path.insert(0, sparsedrive_dir)
            sys.path.insert(0, os.path.join(sparsedrive_dir, 'projects'))

            from mmcv import Config
            from mmdet.models import build_detector

            print(f"[BRIDGE] Ładowanie konfiguracji: {config_path}")
            cfg = Config.fromfile(config_path)

            print("[BRIDGE] Budowanie modelu...")
            self.model = build_detector(cfg.model)
            self.model.eval()
            self.model.cuda()

            if os.path.exists(checkpoint_path):
                print(f"[BRIDGE] Ładowanie checkpointu: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cuda')
                state_dict = checkpoint.get('state_dict', checkpoint)
                # Usuń prefix 'module.' jeśli model był trenowany na wielu GPU
                cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
                missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
                if missing:
                    print(f"[BRIDGE] Brakujące klucze: {len(missing)}")
                if unexpected:
                    print(f"[BRIDGE] Niespodziewane klucze: {len(unexpected)}")
                print(f"[BRIDGE] Model SparseDrive załadowany pomyślnie!")
            else:
                print(f"[BRIDGE] Checkpoint nie znaleziony: {checkpoint_path}")
                print("[BRIDGE] Model będzie używał losowych wag (niska jakość).")

            self.use_fallback = False
            self._model_loaded = True
            print("[BRIDGE] SparseDrive inference engine GOTOWY.")

        except ImportError as e:
            print(f"[BRIDGE] Brak zależności ({e}) — używam fallback percepcji.")
            self.use_fallback = True
        except Exception as e:
            print(f"[BRIDGE] Błąd ładowania modelu: {e}")
            import traceback
            traceback.print_exc()
            print("[BRIDGE] Używam fallback percepcji.")
            self.use_fallback = True

    def process_frame(self, images_6cam: dict, vehicle_state: dict,
                      command: int, cmd_system=None):
        """Przetwarza jedną klatkę: obrazy → model → sterowanie.

        Args:
            images_6cam: dict z 6 obrazami kamer (OpenCV BGR)
            vehicle_state: dict ze stanem pojazdu
            command: int - komenda nawigacyjna
            cmd_system: CommandSystem - do zmiany pasa

        Returns:
            (steering, throttle, brake, results, trajectory_2d, target_speed)
        """
        if self.use_fallback:
            results = self.fallback.process(images_6cam, 0)
        else:
            input_dict = self.input_builder.build(images_6cam, vehicle_state, command)
            img = input_dict['img'].cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model.simple_test(img, **{k: v for k, v in input_dict.items() if k != 'img'})
            results = outputs[0].get('img_bbox', {}) if outputs else {}

        steering, throttle, brake, traj_2d, target_speed = \
            self.control_extractor.extract(results, vehicle_state.get('speed_ms', 10), cmd_system)

        return steering, throttle, brake, results, traj_2d, target_speed

    def check_lane_change_safe(self, results, direction: int):
        """Sprawdza czy zmiana pasa w danym kierunku jest bezpieczna.

        Args:
            results: wyniki z modelu
            direction: -1 (lewo) lub +1 (prawo)

        Returns:
            bool: True jeśli pas jest wolny i istnieje
        """
        if results is None:
            return True  # Fallback: zakładamy że można

        # Sprawdź mapy (czy istnieje pas)
        map_vectors = results.get('vectors', None)
        if map_vectors is not None and len(map_vectors) > 0:
            lane_exists = False
            for i, vec in enumerate(map_vectors):
                label = results['labels'][i]
                if label == 1:  # divider — linia między pasami
                    pts = vec
                    if direction == -1:  # lewy pas
                        if np.any(pts[:, 0] > 2.0):
                            lane_exists = True
                    else:  # prawy pas
                        if np.any(pts[:, 0] < -2.0):
                            lane_exists = True
            if not lane_exists:
                return False

        # Sprawdź detekcje (czy nie ma auta na sąsiednim pasie)
        detections = results.get('boxes_3d', None)
        if detections is not None and len(detections) > 0:
            offset = -LANE_WIDTH if direction == -1 else LANE_WIDTH
            for det in detections:
                if abs(det[0] - offset) < LANE_WIDTH and abs(det[1]) < 20:
                    return False  # Auto na sąsiednim pasie

        return True

"""Most BeamNG ↔ SparseDrive.

Integruje model SparseDrive z symulatorem BeamNG.tech:
- SparseDriveBridge: ładuje model, uruchamia inferencję
- InputBuilder: konwertuje obrazy BeamNG do formatu nuScenes
- ControlExtractor: wyciąga steering/throttle/brake z trajektorii (Pure Pursuit)
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
# INPUT BUILDER — konwersja obrazów BeamNG do formatu modelu
# ==============================================================================

class InputBuilder:
    """Buduje słownik wejściowy w formacie nuScenes dla modelu SparseDrive."""

    def __init__(self):
        self.input_shape = INPUT_SHAPE  # (704, 256) - (w, h)
        self.mean = torch.tensor(IMG_NORM_MEAN, dtype=torch.float32).view(1, 1, 3)
        self.std = torch.tensor(IMG_NORM_STD, dtype=torch.float32).view(1, 1, 3)

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
            'image_wh': torch.tensor([[self.input_shape] * 6], dtype=torch.float32),
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
    """Ładuje model SparseDrive i udostępnia inferencję end-to-end.

    Model SparseDrive (Stage 2): 6 kamer → detekcja 3D + mapy + predykcja ruchu + planowanie.
    """

    def __init__(self, config_path=None, checkpoint_path=None):
        self.model = None
        self.input_builder = InputBuilder()
        self.control_extractor = ControlExtractor()
        self._temporal_queue = deque(maxlen=QUEUE_LENGTH)

        if not config_path or not checkpoint_path:
            raise RuntimeError("SparseDriveBridge wymaga config_path i checkpoint_path.")

        self._try_load_model(config_path, checkpoint_path)

    def _try_load_model(self, config_path, checkpoint_path):
        """Ładuje model SparseDrive z mmcv-full 1.7.1."""
        import sys
        import os

        sparsedrive_dir = os.path.join(os.path.dirname(__file__), 'SparseDrive-main')
        sys.path.insert(0, sparsedrive_dir)
        sys.path.insert(0, os.path.join(sparsedrive_dir, 'projects'))

        # Import MUST happen before build_detector — rejestruje klasę w DETECTORS.
        # mmdet3d_plugin.__init__ importuje datasets które konfliktują z mmdet.
        # Patch registry żeby ignorował duplikaty zamiast crashować.
        from mmcv.utils import Registry
        _orig_register = Registry._register_module
        def _safe_register(self, module, module_name=None, force=False):
            try:
                return _orig_register(self, module, module_name, force)
            except KeyError:
                pass  # już zarejestrowane — ignoruj
        Registry._register_module = _safe_register

        from mmdet3d_plugin.models import SparseDrive as _SD
        from mmcv import Config
        from mmdet.models import build_detector

        print(f"[BRIDGE] Ładowanie konfiguracji: {config_path}")
        cfg = Config.fromfile(config_path)

        # Konfig używa ścieżek relatywnych (data/kmeans/, ckpt/) — trzeba być w katalogu SparseDrive
        _prev_cwd = os.getcwd()
        os.chdir(sparsedrive_dir)
        try:
            print("[BRIDGE] Budowanie modelu...")
            self.model = build_detector(cfg.model)
        finally:
            os.chdir(_prev_cwd)
        self.model.eval()
        self.model.cuda()

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint nie znaleziony: {checkpoint_path}\n"
                f"Pobierz sparsedrive_stage2.pth z:\n"
                f"https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth"
            )

        print(f"[BRIDGE] Ładowanie checkpointu: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        state_dict = checkpoint.get('state_dict', checkpoint)
        cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # Pomijamy anchory — checkpoint ma inne kształty niż nasza konfiguracja
        # (anchory to tylko inicjalne pozycje zapytań, model je adaptuje)
        cleaned = {k: v for k, v in cleaned.items()
                   if 'motion_anchor' not in k and 'plan_anchor' not in k}
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            plan_missing = [k for k in missing if 'plan' in k or 'motion' in k]
            det_missing = [k for k in missing if 'det' in k]
            other_missing = [k for k in missing if 'plan' not in k and 'motion' not in k and 'det' not in k]
            print(f"[BRIDGE] Brakujące klucze: {len(missing)} "
                  f"(det:{len(det_missing)} plan/motion:{len(plan_missing)} other:{len(other_missing)})")
        if unexpected:
            print(f"[BRIDGE] Niespodziewane klucze: {len(unexpected)}")
        print(f"[BRIDGE] Model SparseDrive załadowany pomyślnie!")
        print("[BRIDGE] SparseDrive inference engine GOTOWY.")

    def process_frame(self, images_6cam: dict, vehicle_state: dict,
                      command: int, cmd_system=None):
        """Przetwarza jedną klatkę: obrazy → model SparseDrive → sterowanie.

        Args:
            images_6cam: dict z 6 obrazami kamer (OpenCV BGR)
            vehicle_state: dict ze stanem pojazdu
            command: int - komenda nawigacyjna
            cmd_system: CommandSystem - do zmiany pasa

        Returns:
            (steering, throttle, brake, results, trajectory_2d, target_speed)
        """
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
            results: wyniki z modelu SparseDrive (detekcje 3D + mapy)
            direction: -1 (lewo) lub +1 (prawo)

        Returns:
            bool: True jeśli pas jest wolny i istnieje
        """
        if results is None:
            return True

        # Sprawdź mapy (czy istnieje pas)
        map_vectors = results.get('vectors', None)
        if map_vectors is not None and len(map_vectors) > 0:
            lane_exists = False
            for i, vec in enumerate(map_vectors):
                label = results['labels'][i]
                if label == 1:  # divider
                    pts = vec
                    if direction == -1:
                        if np.any(pts[:, 0] > 2.0):
                            lane_exists = True
                    else:
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
                    return False

        return True

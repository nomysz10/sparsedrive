"""Wizualizacja: 6 kamer (OpenCV), BEV (Matplotlib→OpenCV), Panel sterowania (OpenCV).

Adaptuje kod z SparseDrive:
- bev_render.py → BEVWindow (real-time update, renderowane do numpy i pokazywane przez OpenCV)
- cam_render.py → trajectory_on_front_camera (rzutowanie trajektorii)

Używa backendu 'Agg' dla matplotlib aby uniknąć konfliktów GIL z BeamNGpy.
"""
import math
import time
import cv2
import numpy as np

# Matplotlib z backendem Agg — bezpieczne wątkowe (brak Tkinter/GIL konfliktów)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from config import (
    CAM_ORDER, CAM_TITLES, CAMERA_TILE_SIZE, PANEL_WIDTH, PANEL_HEIGHT,
    BEV_XLIM, BEV_YLIM, BEV_UPDATE_INTERVAL,
    CMD_NAMES, CMD_GO_STRAIGHT, CMD_TURN_LEFT, CMD_TURN_RIGHT,
    STATE_DRIVING_STRAIGHT,
)

# Kolory (z SparseDrive)
COLOR_VECTORS = ['cornflowerblue', 'royalblue', 'slategrey']
CMD_LIST = ['Turn Right', 'Turn Left', 'Go Straight']
SCORE_THRESH = 0.3

# Mapa kolorów dla detekcji (z SparseDrive)
color_mapping = np.asarray([
    [0, 0, 0], [255, 179, 0], [128, 62, 117], [255, 104, 0],
    [166, 189, 215], [193, 0, 32], [206, 162, 98], [129, 112, 102],
    [0, 125, 52], [246, 118, 142], [0, 83, 138], [255, 122, 92],
    [83, 55, 122], [255, 142, 0], [179, 40, 81], [244, 200, 0],
    [127, 24, 13], [147, 170, 0], [89, 51, 21], [241, 58, 19],
    [35, 44, 22], [112, 224, 255], [70, 184, 160], [153, 0, 255],
    [71, 255, 0], [255, 0, 163], [255, 204, 0], [0, 255, 235],
    [255, 0, 235], [255, 0, 122], [255, 245, 0], [10, 190, 212],
    [214, 255, 0], [0, 204, 255], [20, 0, 255], [255, 255, 0],
    [0, 153, 255], [0, 255, 204], [41, 255, 0], [173, 0, 255],
    [0, 245, 255], [71, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [255, 214, 0], [25, 194, 194], [92, 0, 255],
    [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
    [250, 10, 15], [20, 255, 0],
]) / 255


class CameraWindow:
    """Okno OpenCV z 6 kamerami i HUD-em."""

    def __init__(self):
        self.tile_w, self.tile_h = CAMERA_TILE_SIZE
        self.window_w = self.tile_w * 3
        self.window_h = self.tile_h * 2 + 120  # +120 na HUD
        self.window_name = "SparseDrive Cameras"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_w, self.window_h)

    def render(self, images, trajectory_2d, speed_kmh, steering, frame,
               command, status, autopilot_on, lane_change_available):
        """Renderuje widok 6 kamer z HUD i trajektorią na przedniej kamerze.

        Args:
            images: dict z obrazami dla kluczy 'FL', 'F', 'FR', 'BL', 'B', 'BR'
            trajectory_2d: np.array (N,2) - punkty trajektorii w pikselach na przedniej kamerze
            speed_kmh: float
            steering: float
            frame: int
            command: int - bieżąca komenda
            status: str - status jazdy
            autopilot_on: bool
            lane_change_available: dict - {'left': bool, 'right': bool}
        """
        h, w = self.tile_h, self.tile_w
        canvas = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)

        for i, cam_key in enumerate(CAM_ORDER):
            row, col = i // 3, i % 3
            y1, y2 = row * h, (row + 1) * h
            x1, x2 = col * w, (col + 1) * w

            img = images.get(cam_key)
            if img is not None and img.size > 0:
                resized = cv2.resize(img, (w, h))
                canvas[y1:y2, x1:x2] = resized

                # Tytuł kamery
                cv2.putText(canvas, CAM_TITLES[i], (x1 + 8, y1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Rysuj trajektorię na przedniej kamerze
                if cam_key == 'F' and trajectory_2d is not None:
                    for pt in trajectory_2d:
                        px, py = int(pt[0]), int(pt[1])
                        px = x1 + int(px * w / 1600)
                        py = y1 + int(py * h / 900)
                        if 0 <= px < x2 and 0 <= py < y2:
                            cv2.circle(canvas, (px, py), 4, (0, 165, 255), -1)
                    # Połącz kropki linią
                    if len(trajectory_2d) > 1:
                        pts = []
                        for pt in trajectory_2d:
                            px = x1 + int(pt[0] * w / 1600)
                            py = y1 + int(pt[1] * h / 900)
                            pts.append((px, py))
                        pts_array = np.array(pts, dtype=np.int32)
                        cv2.polylines(canvas, [pts_array], False, (0, 140, 220), 2)
            else:
                canvas[y1:y2, x1:x2] = (30, 30, 30)
                cv2.putText(canvas, "NO SIGNAL", (x1 + 60, y1 + h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # HUD - dolny pasek
        hud_y = h * 2 + 5
        cv2.rectangle(canvas, (0, hud_y - 5), (self.window_w, self.window_h), (20, 20, 20), -1)

        # Status autopilota
        ap_color = (0, 255, 0) if autopilot_on else (0, 0, 255)
        ap_text = "AUTOPILOT ON" if autopilot_on else "AUTOPILOT OFF"
        cv2.putText(canvas, ap_text, (15, hud_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ap_color, 2)

        # Prędkość
        cv2.putText(canvas, f"SPEED: {int(speed_kmh)} km/h", (200, hud_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Skręt
        steer_color = (255, 150, 0)
        cv2.putText(canvas, f"STEER: {steering:.2f}", (400, hud_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, steer_color, 1)

        # Komenda
        cmd_name = CMD_NAMES.get(command, 'Unknown')
        cv2.putText(canvas, f"CMD: {cmd_name}", (560, hud_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

        # Status
        status_colors = {
            STATE_DRIVING_STRAIGHT: (0, 255, 0),
            'SKRET': (255, 200, 0),
            'WYPRZEDZANIE': (255, 150, 0),
            'ZMIANA PASA': (200, 200, 0),
            'ZAWROT': (255, 100, 0),
            'HAMOWANIE': (0, 0, 255),
        }
        st_color = (0, 255, 0)
        for key, col in status_colors.items():
            if key in status:
                st_color = col
                break
        cv2.putText(canvas, f"STATUS: {status}", (15, hud_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, st_color, 1)

        # Dostępność pasa
        if lane_change_available:
            left_ok = "TAK" if lane_change_available.get('left') else "NIE"
            right_ok = "TAK" if lane_change_available.get('right') else "NIE"
            cv2.putText(canvas, f"PAS L: {left_ok}  PAS R: {right_ok}",
                        (15, hud_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Ramka
        cv2.putText(canvas, f"FRAME: {frame}", (self.window_w - 150, hud_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        cv2.imshow(self.window_name, canvas)

    def close(self):
        cv2.destroyWindow(self.window_name)


class BEVWindow:
    """Widok z lotu ptaka renderowany przez matplotlib (Agg) i wyświetlany przez OpenCV.

    Używa backendu 'Agg' aby uniknąć konfliktów GIL między Tkinter a BeamNGpy.
    Renderuje figurę do numpy array i pokazuje w oknie OpenCV.
    """

    def __init__(self):
        self.xlim = BEV_XLIM
        self.ylim = BEV_YLIM
        self._last_update_frame = -999
        self._dpi = 80
        self._figsize_inches = (8, 8)
        self._figsize_px = (int(8 * self._dpi), int(8 * self._dpi))  # 640x640
        self.window_name = "BEV - Bird's Eye View"
        self._setup_canvas()

    def _setup_canvas(self):
        """Inicjalizuje figurę matplotlib (Agg backend — bez GUI)."""
        self.fig, self.axes = plt.subplots(
            1, 1, figsize=self._figsize_inches, dpi=self._dpi,
            facecolor='black',
        )
        self.axes.set_xlim(-self.xlim, self.xlim)
        self.axes.set_ylim(-self.ylim, self.ylim)
        self.axes.set_facecolor('black')
        self.axes.axis('off')
        self.fig.tight_layout(pad=0)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self._figsize_px[0], self._figsize_px[1])

    def should_update(self, frame):
        """Sprawdza czy BEV powinien być aktualizowany w tej klatce."""
        return (frame - self._last_update_frame) >= BEV_UPDATE_INTERVAL

    def render(self, frame, results, vehicle_state, command, cmd_system):
        """Aktualizuje widok BEV nowymi danymi.

        Args:
            frame: int - numer klatki
            results: dict - wyniki z modelu SparseDrive (lub None)
            vehicle_state: dict - stan pojazdu z BeamNG
            command: int - bieżąca komenda
            cmd_system: CommandSystem - system nawigacji
        """
        if not self.should_update(frame):
            return
        self._last_update_frame = frame

        self.axes.clear()
        self.axes.set_xlim(-self.xlim, self.xlim)
        self.axes.set_ylim(-self.ylim, self.ylim)
        self.axes.set_facecolor('black')
        self.axes.axis('off')

        # Rysuj ego-pojazd (prostokąt oznaczający auto)
        self._draw_ego_vehicle()

        # Jeśli mamy wyniki modelu — rysuj detekcje, mapy, trajektorie
        if results is not None:
            self._draw_detections(results)
            self._draw_maps(results)
            self._draw_motion_trajectories(results)
            self._draw_planning_trajectory(results)
            self._draw_lane_change_offset(cmd_system)

        # Rysuj komendę
        cmd_name = CMD_NAMES.get(command, 'Go Straight')
        self.axes.text(-38, -38, cmd_name, fontsize=14, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        # Status
        self.axes.text(-38, -35, cmd_system.status, fontsize=10, color='lime',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        # Renderuj figurę do numpy array (Agg → bezpieczne wątkowe)
        bev_image = self._fig_to_numpy()

        # Pokaż przez OpenCV
        cv2.imshow(self.window_name, bev_image)

    def _fig_to_numpy(self):
        """Konwertuje figurę matplotlib na obraz numpy (BGR) dla OpenCV."""
        self.fig.canvas.draw()
        # Pobierz buffer jako RGB
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8)
        # RGBA → BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img_bgr

    def _draw_ego_vehicle(self):
        """Rysuje pojazd ego jako prostokąt."""
        car_w, car_h = 2.0, 4.5
        rect = plt.Rectangle((-car_w / 2, -car_h / 2), car_w, car_h,
                             linewidth=2, edgecolor='lime', facecolor='green',
                             alpha=0.8, zorder=2)
        self.axes.add_patch(rect)
        # Strzałka kierunku
        self.axes.arrow(0, 0, 0, 3.0, head_width=0.5, head_length=0.8,
                        fc='lime', ec='lime', linewidth=2, zorder=3)

    def _draw_detections(self, results):
        """Rysuje wykryte obiekty 3D (boxy)."""
        if 'boxes_3d' not in results:
            return
        bboxes = results['boxes_3d']
        labels = results.get('labels_3d', None)
        scores = results.get('scores_3d', None)

        if isinstance(bboxes, np.ndarray):
            for i in range(len(bboxes)):
                if scores is not None and i < len(scores) and scores[i] < SCORE_THRESH:
                    continue
                color = color_mapping[i % len(color_mapping)]
                cx, cy = bboxes[i][0], bboxes[i][1]
                w, h = bboxes[i][3], bboxes[i][4]
                yaw = bboxes[i][6] if bboxes.shape[1] > 6 else 0

                corners = self._get_box_corners(cx, cy, w, h, yaw)
                self.axes.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
                self.axes.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.2)

    def _draw_maps(self, results):
        """Rysuje wektory mapy (linie pasów, krawędzie drogi, przejścia)."""
        if 'vectors' not in results:
            return
        for i in range(results['scores'].shape[0] if 'scores' in results else 0):
            score = results['scores'][i] if 'scores' in results else 1.0
            if score < SCORE_THRESH:
                continue
            label = results['labels'][i] if 'labels' in results else 0
            color = COLOR_VECTORS[label] if label < len(COLOR_VECTORS) else 'white'
            pts = results['vectors'][i]
            self.axes.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2,
                           marker='o', markersize=3, linestyle='-')

    def _draw_motion_trajectories(self, results):
        """Rysuje przewidywane trajektorie innych pojazdów."""
        if 'trajs_3d' not in results:
            return
        trajs = results['trajs_3d']
        bboxes = results.get('boxes_3d', None)
        if bboxes is None:
            return
        for i in range(min(len(trajs), len(bboxes))):
            score = results.get('scores_3d', np.ones(len(bboxes)))[i]
            if score < SCORE_THRESH:
                continue
            color = color_mapping[results.get('instance_ids', [i])[i] % len(color_mapping)]
            traj = trajs[i]  # (modes, timesteps, 2)
            if traj.ndim == 3:
                traj = traj[0]  # najlepszy mode
            origin = bboxes[i][:2]
            traj_abs = traj.cumsum(axis=0) + origin if hasattr(traj, 'cumsum') else traj
            self.axes.scatter(traj_abs[:, 0], traj_abs[:, 1], c=[color],
                              s=20, alpha=0.7, marker='o')

    def _draw_planning_trajectory(self, results):
        """Rysuje zaplanowaną trajektorię ego."""
        traj = None
        if 'final_planning' in results:
            traj = results['final_planning']
        elif 'planning' in results:
            plan = results['planning']
            if hasattr(plan, 'cpu'):
                plan = plan.cpu().numpy()
            if plan.ndim >= 3:
                plan = plan[0][0]  # pierwsza komenda, pierwszy mode
            traj = plan

        if traj is not None:
            if hasattr(traj, 'cpu'):
                traj = traj.cpu().numpy()
            traj_abs = traj if traj.shape[0] > 1 and abs(traj[0]).sum() < 1 else traj
            # Trajektoria od ego (0,0)
            total_steps = len(traj) * 10
            dot_colors = plt.get_cmap('autumn')(
                np.linspace(0, 1, total_steps))[:, :3]
            for i in range(len(traj)):
                self.axes.scatter(traj[i][1], traj[i][0], c=[dot_colors[i * 10]],
                                  s=40, marker='o', zorder=5)
            # Linia
            self.axes.plot(traj[:, 1], traj[:, 0], color='orange',
                           linewidth=2, linestyle='--', alpha=0.8)

    def _draw_lane_change_offset(self, cmd_system):
        """Wizualizuje offset zmiany pasa na BEV."""
        if cmd_system.is_lane_changing:
            offset = cmd_system.lane_change_offset
            self.axes.arrow(0, 4, offset, 0, head_width=0.3, head_length=0.5,
                            fc='cyan', ec='cyan', linewidth=2, zorder=5)

    @staticmethod
    def _get_box_corners(cx, cy, w, h, yaw):
        """Zwraca narożniki prostokąta obróconego o yaw."""
        corners = np.array([
            [-w / 2, -h / 2],
            [w / 2, -h / 2],
            [w / 2, h / 2],
            [-w / 2, h / 2],
            [-w / 2, -h / 2],
        ])
        rot = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)],
        ])
        rotated = corners @ rot.T
        return rotated + np.array([cx, cy])

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
        cv2.destroyWindow(self.window_name)


class ControlPanel:
    """Panel sterowania (OpenCV) z przyciskami i suwakami."""

    def __init__(self):
        self.w = PANEL_WIDTH
        self.h = PANEL_HEIGHT
        self.window_name = "Control Panel"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.w, self.h)

        # Stworzenie trackbara dla autopilota
        cv2.createTrackbar('Autopilot', self.window_name, 1, 1, lambda x: None)

        # Callbacki myszy
        self._buttons = {}  # {nazwa: (x, y, w, h)}
        self._last_click = None
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # Stan przycisków
        self._btn_states = {}
        self._pressed_button = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._last_click = (x, y)
            for name, (bx, by, bw, bh) in self._buttons.items():
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._pressed_button = name
                    break

    def render(self, speed_kmh, command, cmd_system, auto_on):
        """Renderuje panel sterowania.

        Returns:
            str or None: nazwa wciśniętego przycisku
        """
        canvas = np.ones((self.h, self.w, 3), dtype=np.uint8) * 40
        pressed = self._pressed_button
        self._pressed_button = None

        # Tytuł
        cv2.putText(canvas, "CONTROL PANEL", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(canvas, (10, 40), (self.w - 10, 40), (100, 100, 100), 1)

        # Autopilot
        ap_color = (0, 255, 0) if auto_on else (0, 0, 255)
        ap_text = "ON" if auto_on else "OFF"
        cv2.putText(canvas, f"AUTOPILOT: {ap_text}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ap_color, 2)
        cv2.putText(canvas, "[SPACE] to toggle", (15, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Prędkość
        cv2.putText(canvas, f"SPEED: {int(speed_kmh)} km/h", (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Status
        cv2.putText(canvas, f"STATUS:", (15, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        status_color = (0, 255, 0)
        if 'HAMOWANIE' in cmd_system.status:
            status_color = (0, 0, 255)
        elif 'SKRET' in cmd_system.status or 'WYPRZEDZ' in cmd_system.status:
            status_color = (255, 200, 0)
        cv2.putText(canvas, cmd_system.status, (15, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

        # Separator
        cv2.line(canvas, (10, 200), (self.w - 10, 200), (100, 100, 100), 1)

        # Przyciski nawigacji
        cv2.putText(canvas, "NAWIGACJA:", (15, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        btn_y = 250
        btn_h = 45
        btn_w = 110
        gap = 10

        # Rząd 1: Turn Left | Go Straight | Turn Right
        row1_btns = [
            ('TURN LEFT', 15, btn_y, btn_w, btn_h, (255, 140, 0)),
            ('GO STRAIGHT', 15 + btn_w + gap, btn_y, btn_w, btn_h, (0, 180, 0)),
            ('TURN RIGHT', 15 + (btn_w + gap) * 2 + 5, btn_y, btn_w + 5, btn_h, (255, 140, 0)),
        ]

        for name, bx, by, bw, bh, color in row1_btns:
            highlight = (command == CMD_TURN_LEFT and 'LEFT' in name) or \
                        (command == CMD_TURN_RIGHT and 'RIGHT' in name) or \
                        (command == CMD_GO_STRAIGHT and 'STRAIGHT' in name)
            btn_color = (0, 255, 0) if highlight else color
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), btn_color, -1 if highlight else 2)
            cv2.putText(canvas, name, (bx + 5, by + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            self._buttons[name] = (bx, by, bw, bh)

        # Rząd 2: Lane Change
        btn_y2 = btn_y + btn_h + 15
        lane_btns = [
            ('LANE <<', 15, btn_y2, btn_w, btn_h, (200, 200, 0)),
            ('LANE >>', 15 + btn_w + gap, btn_y2, btn_w, btn_h, (200, 200, 0)),
        ]
        for name, bx, by, bw, bh, color in lane_btns:
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(canvas, name, (bx + 8, by + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self._buttons[name] = (bx, by, bw, bh)

        # Rząd 3: U-Turn
        btn_y3 = btn_y2 + btn_h + 15
        cv2.rectangle(canvas, (15, btn_y3), (15 + btn_w, btn_y3 + btn_h), (255, 100, 100), 2)
        cv2.putText(canvas, 'U-TURN', (25, btn_y3 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self._buttons['U-TURN'] = (15, btn_y3, btn_w, btn_h)

        # Separator
        cv2.line(canvas, (10, 430), (self.w - 10, 430), (100, 100, 100), 1)

        # Wyjście
        cv2.putText(canvas, "Press Q in camera window to exit", (15, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(canvas, "Press R to reset command to STRAIGHT", (15, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(canvas, "Press E for emergency brake", (15, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow(self.window_name, canvas)
        return pressed

    def close(self):
        cv2.destroyWindow(self.window_name)

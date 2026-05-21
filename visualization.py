"""Wizualizacja: wszystkie widoki w jednym oknie OpenCV.

Layout:
┌─────────────────────────────────────────────┐
│ FL (360x200) │ F (360x200) │ FR (360x200)  │  ← 6 kamer 3x2
│ BL (360x200) │ B (360x200) │ BR (360x200)  │
├──────────────────┬──────────────────────────┤
│                  │  CONTROL PANEL           │
│  BEV (500x500)   │  - AUTOPILOT ON/OFF      │
│                  │  - SPEED                 │
│                  │  - STATUS                │
│                  │  - NAWIGACJA             │
│                  │  - LANE CHANGE           │
│                  │  - U-TURN                │
├──────────────────┴──────────────────────────┤
│ HUD: SPEED | STEER | CMD | STATUS | LANES   │
└─────────────────────────────────────────────┘
"""
import math
import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    CAM_ORDER, CAM_TITLES,
    BEV_XLIM, BEV_YLIM, BEV_UPDATE_INTERVAL,
    CMD_NAMES, CMD_GO_STRAIGHT, CMD_TURN_LEFT, CMD_TURN_RIGHT,
    STATE_DRIVING_STRAIGHT, PANEL_WIDTH, PANEL_HEIGHT,
)

# Kolory SparseDrive
COLOR_VECTORS = ['cornflowerblue', 'royalblue', 'slategrey']
SCORE_THRESH = 0.3

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


# ==============================================================================
# UNIFIED WINDOW
# ==============================================================================

class SparseDriveUI:
    """Jedno okno OpenCV łączące kamery, BEV i panel sterowania."""

    # Stałe layoutu
    CAM_TILE_W, CAM_TILE_H = 360, 200
    BEV_SIZE = 500
    PANEL_W = 350
    HUD_H = 70

    @property
    def cam_area_w(self):
        return self.CAM_TILE_W * 3  # 1080

    @property
    def cam_area_h(self):
        return self.CAM_TILE_H * 2  # 400

    @property
    def bottom_area_h(self):
        return max(self.BEV_SIZE, 500)  # BEV 500 + przyciski

    @property
    def total_w(self):
        return self.cam_area_w  # 1080

    @property
    def total_h(self):
        return self.cam_area_h + self.bottom_area_h + self.HUD_H

    def __init__(self):
        self.window_name = "SparseDrive Autonomous"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.total_w, self.total_h)

        # BEV – matplotlib figura
        self._last_bev_frame = -999
        self._bev_dpi = 80
        self._bev_figsize = (self.BEV_SIZE / self._bev_dpi, self.BEV_SIZE / self._bev_dpi)
        self._bev_fig, self._bev_ax = plt.subplots(
            1, 1, figsize=self._bev_figsize, dpi=self._bev_dpi, facecolor='black',
        )
        self._bev_ax.set_xlim(-BEV_XLIM, BEV_XLIM)
        self._bev_ax.set_ylim(-BEV_YLIM, BEV_YLIM)
        self._bev_ax.set_facecolor('black')
        self._bev_ax.axis('off')
        self._bev_fig.tight_layout(pad=0)

        # Przyciski panelu
        self._buttons = {}
        self._pressed_button = None
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (bx, by, bw, bh) in self._buttons.items():
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._pressed_button = name
                    break

    # ==========================================================================
    # GŁÓWNY RENDER
    # ==========================================================================

    def render(self, images, trajectory_2d, speed_kmh, steering, frame,
               command, status, autopilot_on, lane_change_available,
               results, vehicle_state, cmd_system):
        """Renderuje całe okno: kamery + BEV + panel + HUD.

        Returns:
            str or None: nazwa wciśniętego przycisku
        """
        canvas = np.zeros((self.total_h, self.total_w, 3), dtype=np.uint8)
        self._buttons.clear()

        # 1. Kamery (góra)
        self._render_cameras(canvas, images, trajectory_2d)

        # 2. BEV (dół-lewo)
        self._render_bev(canvas, frame, results, command, cmd_system)

        # 3. Panel sterowania (dół-prawo)
        self._render_panel(canvas, speed_kmh, command, cmd_system, autopilot_on)

        # 4. HUD (dół)
        self._render_hud(canvas, speed_kmh, steering, frame, command, status,
                         autopilot_on, lane_change_available)

        cv2.imshow(self.window_name, canvas)

        pressed = self._pressed_button
        self._pressed_button = None
        return pressed

    # ==========================================================================
    # KAMERY
    # ==========================================================================

    def _render_cameras(self, canvas, images, trajectory_2d):
        tw, th = self.CAM_TILE_W, self.CAM_TILE_H
        for i, cam_key in enumerate(CAM_ORDER):
            row, col = i // 3, i % 3
            y1, y2 = row * th, (row + 1) * th
            x1, x2 = col * tw, (col + 1) * tw

            img = images.get(cam_key)
            if img is not None and img.size > 0:
                resized = cv2.resize(img, (tw, th))
                canvas[y1:y2, x1:x2] = resized

                # Etykieta kamery
                cv2.putText(canvas, CAM_TITLES[i], (x1 + 8, y1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Trajektoria na przedniej kamerze (jak w oryginalnym SparseDrive)
                if cam_key == 'F' and trajectory_2d is not None and len(trajectory_2d) > 1:
                    pts = []
                    for pt in trajectory_2d:
                        px = x1 + int(pt[0] * tw / 1600)
                        py = y1 + int(pt[1] * th / 900)
                        pts.append((px, py))
                    pts_array = np.array(pts, dtype=np.int32)
                    cv2.polylines(canvas, [pts_array], False, (0, 165, 255), 2)
                    for pt in pts:
                        cv2.circle(canvas, pt, 3, (0, 140, 255), -1)
            else:
                canvas[y1:y2, x1:x2] = (30, 30, 30)
                cv2.putText(canvas, "NO SIGNAL", (x1 + tw // 3, y1 + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # ==========================================================================
    # BEV
    # ==========================================================================

    def _render_bev(self, canvas, frame, results, command, cmd_system):
        """BEV w lewym dolnym rogu."""
        x_off, y_off = 0, self.cam_area_h

        if (frame - self._last_bev_frame) < BEV_UPDATE_INTERVAL:
            return
        self._last_bev_frame = frame

        ax = self._bev_ax
        ax.clear()
        ax.set_xlim(-BEV_XLIM, BEV_XLIM)
        ax.set_ylim(-BEV_YLIM, BEV_YLIM)
        ax.set_facecolor('black')
        ax.axis('off')

        # Ego
        car_w, car_h = 2.0, 4.5
        rect = plt.Rectangle((-car_w / 2, -car_h / 2), car_w, car_h,
                             linewidth=2, edgecolor='lime', facecolor='green',
                             alpha=0.8, zorder=2)
        ax.add_patch(rect)
        ax.arrow(0, 0, 0, 3.0, head_width=0.5, head_length=0.8,
                fc='lime', ec='lime', linewidth=2, zorder=3)

        if results is not None:
            self._draw_detections(ax, results)
            self._draw_maps(ax, results)
            self._draw_planning(ax, results)

        # Komenda + status
        cmd_name = CMD_NAMES.get(command, 'Go Straight')
        ax.text(-38, -38, cmd_name, fontsize=10, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.text(-38, -35, cmd_system.status, fontsize=8, color='lime',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        # Konwersja matplotlib → OpenCV BGR
        self._bev_fig.canvas.draw()
        buf = self._bev_fig.canvas.buffer_rgba()
        bev_img = np.asarray(buf, dtype=np.uint8)
        bev_bgr = cv2.cvtColor(bev_img, cv2.COLOR_RGBA2BGR)
        bev_resized = cv2.resize(bev_bgr, (self.BEV_SIZE, self.BEV_SIZE))

        # Umieść na canvasie
        h, w = bev_resized.shape[:2]
        canvas[y_off:y_off + h, x_off:x_off + w] = bev_resized

    def _draw_detections(self, ax, results):
        if 'boxes_3d' not in results:
            return
        bboxes = results['boxes_3d']
        scores = results.get('scores_3d', None)
        if not isinstance(bboxes, np.ndarray) or len(bboxes) == 0:
            return
        for i in range(len(bboxes)):
            if scores is not None and i < len(scores) and scores[i] < SCORE_THRESH:
                continue
            color = color_mapping[i % len(color_mapping)]
            cx, cy = bboxes[i][0], bboxes[i][1]
            w, h = bboxes[i][3], bboxes[i][4]
            yaw = bboxes[i][6] if bboxes.shape[1] > 6 else 0
            corners = np.array([
                [-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2], [-w / 2, -h / 2],
            ])
            rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            rotated = corners @ rot.T + np.array([cx, cy])
            ax.plot(rotated[:, 0], rotated[:, 1], color=color, linewidth=2)
            ax.fill(rotated[:, 0], rotated[:, 1], color=color, alpha=0.2)

    def _draw_maps(self, ax, results):
        if 'vectors' not in results:
            return
        for i in range(results.get('scores', np.array([])).shape[0]):
            score = results['scores'][i]
            if score < SCORE_THRESH:
                continue
            label = results['labels'][i]
            color = COLOR_VECTORS[label] if label < len(COLOR_VECTORS) else 'white'
            pts = results['vectors'][i]
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5, marker='o', markersize=2)

    def _draw_planning(self, ax, results):
        traj = results.get('final_planning') or results.get('planning')
        if traj is None:
            return
        if hasattr(traj, 'cpu'):
            traj = traj.cpu().numpy()
        # Obsługa różnych kształtów: [batch, cmd, mode, pts, xy] → [N, 2]
        if isinstance(traj, np.ndarray):
            if traj.ndim >= 4:
                traj = traj[0][0][0]  # batch=0, cmd=0, mode=0
            elif traj.ndim == 3:
                traj = traj[0][0]
            elif traj.ndim == 2 and traj.shape[0] > 6:
                traj = traj[:6]
        if traj.ndim != 2 or traj.shape[1] < 2:
            return
        ax.plot(traj[:, 1], traj[:, 0], color='orange', linewidth=2, linestyle='--', alpha=0.9)
        colors = plt.get_cmap('autumn')(np.linspace(0, 1, len(traj)))[:, :3]
        for i in range(len(traj)):
            ax.scatter(traj[i][1], traj[i][0], c=[colors[i]], s=30, marker='o', zorder=5)

    # ==========================================================================
    # PANEL STEROWANIA
    # ==========================================================================

    def _render_panel(self, canvas, speed_kmh, command, cmd_system, auto_on):
        x_off = self.BEV_SIZE  # zaczyna się po BEV
        y_off = self.cam_area_h
        pw = self.total_w - x_off  # szerokość panelu (~230px)
        ph = self.bottom_area_h

        # Tło panelu
        canvas[y_off:y_off + ph, x_off:self.total_w] = (30, 30, 30)

        x = x_off + 10
        y = y_off + 10
        line_h = 28
        btn_h = 38
        btn_w = pw - 30

        # Tytuł
        cv2.putText(canvas, "CONTROLS", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40

        # Autopilot
        ap_color = (0, 255, 0) if auto_on else (0, 0, 255)
        cv2.putText(canvas, f"AP: {'ON' if auto_on else 'OFF'}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ap_color, 2)
        cv2.putText(canvas, "[SPACE]", (x + 130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y += line_h + 5

        # Prędkość
        cv2.putText(canvas, f"{int(speed_kmh)} km/h", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        y += line_h + 5

        # Status
        st_color = (0, 255, 0)
        if 'HAMOWANIE' in cmd_system.status:
            st_color = (0, 0, 255)
        elif 'SKRET' in cmd_system.status or 'WYPRZEDZ' in cmd_system.status:
            st_color = (255, 200, 0)
        cv2.putText(canvas, cmd_system.status, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, st_color, 1)
        y += line_h

        # Separator
        cv2.line(canvas, (x_off + 5, y), (self.total_w - 5, y), (80, 80, 80), 1)
        y += 10

        # Nawigacja
        cv2.putText(canvas, "NAVIGATION", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += line_h

        nav_btns = ['TURN LEFT', 'GO STRAIGHT', 'TURN RIGHT']
        nav_colors = [(255, 140, 0), (0, 180, 0), (255, 140, 0)]
        for name, color in zip(nav_btns, nav_colors):
            highlight = (command == CMD_TURN_LEFT and 'LEFT' in name) or \
                       (command == CMD_TURN_RIGHT and 'RIGHT' in name) or \
                       (command == CMD_GO_STRAIGHT and 'STRAIGHT' in name)
            bc = (0, 255, 0) if highlight else color
            cv2.rectangle(canvas, (x, y), (x + btn_w, y + btn_h), bc, -1 if highlight else 2)
            cv2.putText(canvas, name, (x + 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self._buttons[name] = (x, y, btn_w, btn_h)
            y += btn_h + 8

        y += 5

        # Zmiana pasa
        cv2.putText(canvas, "LANE CHANGE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += line_h

        for name in ['LANE <<', 'LANE >>']:
            cv2.rectangle(canvas, (x, y), (x + btn_w, y + btn_h), (200, 200, 0), 2)
            cv2.putText(canvas, name, (x + 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self._buttons[name] = (x, y, btn_w, btn_h)
            y += btn_h + 8

        # U-Turn
        cv2.rectangle(canvas, (x, y), (x + btn_w, y + btn_h), (255, 100, 100), 2)
        cv2.putText(canvas, 'U-TURN', (x + 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self._buttons['U-TURN'] = (x, y, btn_w, btn_h)

    # ==========================================================================
    # HUD
    # ==========================================================================

    def _render_hud(self, canvas, speed_kmh, steering, frame, command, status,
                    autopilot_on, lane_change_available):
        y = self.cam_area_h + self.bottom_area_h
        h = self.HUD_H

        cv2.rectangle(canvas, (0, y), (self.total_w, y + h), (15, 15, 15), -1)

        # Autopilot
        ap_color = (0, 255, 0) if autopilot_on else (0, 0, 255)
        cv2.putText(canvas, "AUTOPILOT ON" if autopilot_on else "AUTOPILOT OFF",
                    (15, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ap_color, 2)

        # Prędkość
        cv2.putText(canvas, f"{int(speed_kmh)} km/h", (220, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Skręt
        cv2.putText(canvas, f"STEER: {steering:.2f}", (380, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 50), 1)

        # Komenda
        cmd_name = CMD_NAMES.get(command, 'Unknown')
        cv2.putText(canvas, f"CMD: {cmd_name}", (540, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Status
        st_color = (0, 255, 0)
        if 'HAMOWANIE' in status:
            st_color = (0, 0, 255)
        elif 'SKRET' in status or 'WYPRZEDZ' in status:
            st_color = (255, 200, 0)
        cv2.putText(canvas, status, (540, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, st_color, 1)

        # Dostępność pasów
        if lane_change_available:
            left_ok = "OK" if lane_change_available.get('left') else "NO"
            right_ok = "OK" if lane_change_available.get('right') else "NO"
            cv2.putText(canvas, f"L:{left_ok} R:{right_ok}",
                        (740, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Ramka
        cv2.putText(canvas, f"F:{frame}", (self.total_w - 100, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    def close(self):
        if self._bev_fig is not None:
            plt.close(self._bev_fig)
        cv2.destroyWindow(self.window_name)

"""
SparseDrive + BeamNG.tech — autonomiczna jazda z wizualizacją.

Uruchomienie:
    python main.py

Sterowanie:
    Q - wyjście
    SPACJA - włącz/wyłącz autopilota
    E - hamulec awaryjny
    R - reset komendy (Go Straight)
    1 - Turn Left
    2 - Go Straight
    3 - Turn Right
    4 - Zmiana pasa w lewo
    5 - Zmiana pasa w prawo
    6 - Zawróć
"""
import time
import math
import sys
import os
import cv2
import numpy as np
from beamngpy import BeamNGpy, Vehicle, Scenario
from beamngpy.sensors import Camera

from config import (
    BEAMNG_HOME, BEAMNG_HOST, BEAMNG_PORT, MAP_NAME,
    VEHICLE_MODEL, VEHICLE_LICENCE, START_POS, START_YAW_DEG,
    CAM_RESOLUTION, CAM_UPDATE_TIME, CAM_CONFIGS, CAM_ORDER,
    SPARSEDRIVE_CONFIG, SPARSEDRIVE_CHECKPOINT,
    CMD_GO_STRAIGHT, CMD_TURN_LEFT, CMD_TURN_RIGHT,
    STATE_DRIVING_STRAIGHT,
)
from bridge import SparseDriveBridge
from navigation import CommandSystem
from visualization import CameraWindow, BEVWindow, ControlPanel


# ==============================================================================
# INICJALIZACJA BEAMNG
# ==============================================================================

def init_beamng():
    """Łączy się z BeamNG.tech i tworzy scenariusz z pojazdem."""
    print("[SYSTEM] Nawiązywanie połączenia z BeamNG.tech...")
    bng = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT, home=BEAMNG_HOME)
    bng.open()
    print("[SYSTEM] Połączono.")

    vehicle = Vehicle('ego_vehicle', model=VEHICLE_MODEL, licence=VEHICLE_LICENCE)
    scenario = Scenario(MAP_NAME, 'sparsedrive_autonomous')

    yaw_rad = math.radians(START_YAW_DEG) + math.pi
    q_z = math.sin(yaw_rad / 2.0)
    q_w = math.cos(yaw_rad / 2.0)

    scenario.add_vehicle(vehicle, pos=START_POS, rot_quat=(0, 0, q_z, q_w))
    scenario.make(bng)

    print("[SYSTEM] Wczytywanie mapy i scenariusza...")
    bng.scenario.load(scenario)
    time.sleep(3)
    bng.scenario.start()
    time.sleep(2)

    return bng, vehicle


# ==============================================================================
# INICJALIZACJA KAMER
# ==============================================================================

def init_cameras(bng, vehicle):
    """Inicjalizuje 6 kamer wokół pojazdu w rozdzielczości 1600x900."""
    print("[SYSTEM] Inicjalizacja 6 kamer...")
    cam_args = {
        'resolution': CAM_RESOLUTION,
        'requested_update_time': CAM_UPDATE_TIME,
        'is_streaming': True,
        'is_using_shared_memory': True,
    }

    cameras_dict = {}
    for key, cfg in CAM_CONFIGS.items():
        cameras_dict[key] = Camera(
            cfg['name'], bng, vehicle,
            pos=cfg['pos'], dir=cfg['dir'],
            **cam_args
        )
        print(f"  {key}: {cfg['name']} → pos={cfg['pos']}, dir={cfg['dir']}")

    return cameras_dict


# ==============================================================================
# ODCZYT STANU POJAZDU
# ==============================================================================

def read_vehicle_state(vehicle):
    """Odczytuje stan pojazdu z BeamNG."""
    try:
        vehicle.sensors.poll()
        vel = vehicle.state.get('vel', (0, 0, 0))
        speed_ms = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        speed_kmh = speed_ms * 3.6

        pos = vehicle.state.get('pos', (0, 0, 0))
        rot = vehicle.state.get('rot', (0, 0, 0, 1))

        # Wyciągnij yaw z kwaternionu
        qx, qy, qz, qw = rot if len(rot) == 4 else (0, 0, rot[2], rot[3] if len(rot) > 3 else 1)
        sin_yaw = 2 * (qw * qz + qx * qy)
        cos_yaw = 1 - 2 * (qz * qz + qy * qy)
        yaw = math.atan2(sin_yaw, cos_yaw)

        return {
            'speed_ms': speed_ms,
            'speed_kmh': speed_kmh,
            'pos': pos,
            'yaw': yaw,
            'vel': vel,
            'acc': vehicle.state.get('acc', (0, 0, 0)),
            'yaw_rate': vehicle.state.get('yaw_rate', 0),
        }
    except Exception as e:
        return {
            'speed_ms': 0, 'speed_kmh': 0,
            'pos': (0, 0, 0), 'yaw': 0,
            'vel': (0, 0, 0), 'acc': (0, 0, 0), 'yaw_rate': 0,
        }


# ==============================================================================
# POBRANIE OBRAZÓW Z KAMER
# ==============================================================================

def capture_cameras(cameras_dict):
    """Pobiera obrazy ze wszystkich kamer."""
    images = {}
    for key in CAM_ORDER:
        cam = cameras_dict.get(key)
        if cam is None:
            images[key] = np.zeros((900, 1600, 3), dtype=np.uint8)
            continue
        try:
            stream_data = cam.stream()
            if stream_data is not None and 'colour' in stream_data:
                img_pil = stream_data['colour']
                img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                images[key] = img_bgr
            else:
                images[key] = np.zeros((900, 1600, 3), dtype=np.uint8)
        except Exception:
            images[key] = np.zeros((900, 1600, 3), dtype=np.uint8)
    return images


# ==============================================================================
# OBSŁUGA KLAWIATURY
# ==============================================================================

def handle_keyboard(key, cmd_system, autopilot_on):
    """Obsługuje klawisze sterujące.

    Returns:
        (cmd_system, autopilot_on, quit_flag)
    """
    quit_flag = False

    if key == ord('q') or key == ord('Q'):
        quit_flag = True
    elif key == ord(' '):  # Spacja = toggle autopilot
        autopilot_on = not autopilot_on
        print(f"[CONTROL] Autopilot: {'ON' if autopilot_on else 'OFF'}")
    elif key == ord('e') or key == ord('E'):
        if cmd_system.emergency_brake:
            cmd_system.set_emergency_brake(False)
            print("[CONTROL] Hamulec awaryjny zwolniony")
        else:
            cmd_system.set_emergency_brake(True)
            print("[CONTROL] HAMOWANIE AWARYJNE!")
    elif key == ord('r') or key == ord('R'):
        cmd_system.set_command(CMD_GO_STRAIGHT)
        print("[CONTROL] Reset: Go Straight")
    elif key == ord('1'):
        cmd_system.set_command(CMD_TURN_LEFT)
        print("[CONTROL] Komenda: Turn Left")
    elif key == ord('2'):
        cmd_system.set_command(CMD_GO_STRAIGHT)
        print("[CONTROL] Komenda: Go Straight")
    elif key == ord('3'):
        cmd_system.set_command(CMD_TURN_RIGHT)
        print("[CONTROL] Komenda: Turn Right")
    elif key == ord('4'):
        cmd_system.start_lane_change(-1)
        print("[CONTROL] Zmiana pasa w LEWO")
    elif key == ord('5'):
        cmd_system.start_lane_change(+1)
        print("[CONTROL] Zmiana pasa w PRAWO")
    elif key == ord('6'):
        # Zawrót — potrzebuje aktualnego yaw (podane w pętli)
        print("[CONTROL] Zawrót zainicjowany")
        # Yaw będzie ustawiony w pętli głównej

    return cmd_system, autopilot_on, quit_flag


# ==============================================================================
# PĘTLA GŁÓWNA
# ==============================================================================

def main():
    print("=" * 60)
    print("  SPARSEDRIVE + BEAMNG — AUTONOMOUS DRIVING")
    print("=" * 60)

    # 1. Inicjalizacja BeamNG
    bng, vehicle = init_beamng()

    # 2. Inicjalizacja kamer
    cameras_dict = init_cameras(bng, vehicle)

    # 3. Inicjalizacja mostu SparseDrive
    config_path = os.path.join(os.path.dirname(__file__), SPARSEDRIVE_CONFIG)
    checkpoint_path = os.path.join(os.path.dirname(__file__), SPARSEDRIVE_CHECKPOINT)
    bridge = SparseDriveBridge(config_path, checkpoint_path)
    print(f"[SYSTEM] Bridge gotowy (fallback={'TAK' if bridge.use_fallback else 'NIE'})")

    # 4. Inicjalizacja nawigacji
    cmd_system = CommandSystem()

    # 5. Inicjalizacja wizualizacji
    cam_window = CameraWindow()
    bev_window = BEVWindow()
    ctrl_panel = ControlPanel()

    # 6. Stan
    autopilot_on = False  # Start z wyłączonym autopilotem — użytkownik włącza spacją
    frame = 0
    steering = 0.0
    throttle = 0.0
    brake = 0.0
    target_speed = 0.0
    trajectory_2d = None
    results = None
    lane_change_available = {'left': True, 'right': True}

    print("\n[SYSTEM] Start pętli głównej.")
    print("  SPACJA = toggle autopilot | 1/2/3 = nawigacja | 4/5 = zmiana pasa")
    print("  6 = zawrót | E = hamulec awaryjny | Q = wyjście")
    print()

    try:
        while True:
            loop_start = time.time()

            # --- Krok fizyki ---
            bng.step(1)

            # --- Pobranie obrazów ---
            images = capture_cameras(cameras_dict)

            # --- Odczyt stanu pojazdu ---
            vehicle_state = read_vehicle_state(vehicle)

            if autopilot_on:
                # --- Inferencja modelu / fallback ---
                steering, throttle, brake, results, trajectory_2d, target_speed = \
                    bridge.process_frame(images, vehicle_state,
                                         cmd_system.current_command, cmd_system)

                # --- Sprawdzenie dostępności pasów ---
                if not bridge.use_fallback:
                    lane_left_free = bridge.check_lane_change_safe(results, -1)
                    lane_right_free = bridge.check_lane_change_safe(results, +1)
                else:
                    lane_left_free = results.get('lane_available', {}).get('left', True)
                    lane_right_free = results.get('lane_available', {}).get('right', True)
                lane_change_available = {'left': lane_left_free, 'right': lane_right_free}

                # --- Aktualizacja zawracania ---
                cmd_system.update_uturn(vehicle_state['yaw'])
            else:
                steering = 0.0
                throttle = 0.0
                brake = 0.0

            # --- Wysłanie sterowania ---
            vehicle.control(throttle=throttle, steering=steering, brake=brake)

            # --- Wizualizacja ---
            cam_window.render(
                images, trajectory_2d, vehicle_state['speed_kmh'],
                steering, frame, cmd_system.current_command,
                cmd_system.status, autopilot_on, lane_change_available,
            )
            bev_window.render(frame, results, vehicle_state,
                              cmd_system.current_command, cmd_system)
            pressed_btn = ctrl_panel.render(
                vehicle_state['speed_kmh'], cmd_system.current_command,
                cmd_system, autopilot_on,
            )

            # --- Obsługa przycisków panelu sterowania ---
            if pressed_btn:
                if pressed_btn == 'TURN LEFT':
                    cmd_system.set_command(CMD_TURN_LEFT)
                elif pressed_btn == 'GO STRAIGHT':
                    cmd_system.set_command(CMD_GO_STRAIGHT)
                elif pressed_btn == 'TURN RIGHT':
                    cmd_system.set_command(CMD_TURN_RIGHT)
                elif pressed_btn == 'LANE <<':
                    if lane_change_available['left']:
                        cmd_system.start_lane_change(-1)
                        print("[CONTROL] Zmiana pasa w LEWO")
                    else:
                        print("[CONTROL] Lewy pas NIEDOSTĘPNY!")
                elif pressed_btn == 'LANE >>':
                    if lane_change_available['right']:
                        cmd_system.start_lane_change(+1)
                        print("[CONTROL] Zmiana pasa w PRAWO")
                    else:
                        print("[CONTROL] Prawy pas NIEDOSTĘPNY!")
                elif pressed_btn == 'U-TURN':
                    cmd_system.start_uturn(vehicle_state['yaw'])
                    print("[CONTROL] Zawrót rozpoczęty")

            # --- Klawisze ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('6'):
                cmd_system.start_uturn(vehicle_state['yaw'])
                print("[CONTROL] Zawrót rozpoczęty")

            cmd_system, autopilot_on, quit_flag = handle_keyboard(
                key, cmd_system, autopilot_on)

            if quit_flag:
                break

            # --- Log co 50 klatek ---
            frame += 1
            if frame % 50 == 0:
                print(f"[AUTONOMY] Klatka: {frame} | "
                      f"Prędkość: {int(vehicle_state['speed_kmh'])} km/h | "
                      f"Skręt: {steering:.3f} | "
                      f"Status: {cmd_system.status}")

            # --- Kontrola FPS (max ~30) ---
            elapsed = time.time() - loop_start
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Przerwano ręcznie (Ctrl+C).")
    except Exception as e:
        print(f"\n[SYSTEM] Błąd krytyczny: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[SYSTEM] Zamykanie...")
        cam_window.close()
        bev_window.close()
        ctrl_panel.close()
        cv2.destroyAllWindows()
        bng.close()
        print("[SYSTEM] Koniec.")


if __name__ == '__main__':
    main()

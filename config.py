"""Konfiguracja dla autonomicznej jazdy SparseDrive + BeamNG."""
import math
import numpy as np

# ==============================================================================
# BEAMNG CONNECTION
# ==============================================================================
BEAMNG_HOME = 'G:\\Gry\\BeamNG.tech.v0.38.5.0'
BEAMNG_HOST = 'localhost'
BEAMNG_PORT = 64256
MAP_NAME = 'west_coast_usa'

# ==============================================================================
# POJAZD I KAMERY
# ==============================================================================
VEHICLE_MODEL = 'etki'
VEHICLE_LICENCE = 'AUTONOMY'

# Pozycja startowa i orientacja
START_POS = (-1045.877197, -518.1680298, 101.8693466)
START_YAW_DEG = -135.678

# Rozdzielczość kamer (nuScenes standard)
CAM_RESOLUTION = (1600, 900)
CAM_UPDATE_TIME = 0.01

# Pozycje 6 kamer względem pojazdu (x=lewo/prawo, y=przód/tył, z=góra/dół)
CAM_CONFIGS = {
    'F':   dict(pos=(0, -1.6, 1.3),  dir=(0, -1, 0),  name='cam_F'),
    'FL':  dict(pos=(0.8, -1.2, 1.3),  dir=(1, -1, 0),   name='cam_FL'),
    'FR':  dict(pos=(-0.8, -1.2, 1.3), dir=(-1, -1, 0),  name='cam_FR'),
    'B':   dict(pos=(0, 1.6, 1.3),   dir=(0, 1, 0),    name='cam_B'),
    'BL':  dict(pos=(0.8, 1.2, 1.3),   dir=(1, 1, 0),    name='cam_BL'),
    'BR':  dict(pos=(-0.8, 1.2, 1.3),  dir=(-1, 1, 0),   name='cam_BR'),
}

# Kolejność wyświetlania w widoku 3x2 (dopasowana do SparseDrive cam_render.py)
CAM_ORDER = ['FL', 'F', 'FR', 'BL', 'B', 'BR']
CAM_TITLES = ['FRONT LEFT', 'FRONT (TRAJECTORY)', 'FRONT RIGHT', 'BACK LEFT', 'BACK', 'BACK RIGHT']

# ==============================================================================
# KALIBRACJA KAMER (syntetyczna, dopasowana do BeamNG)
# ==============================================================================
CAM_INTRINSIC = np.array([
    [800, 0, 800],   # fx, 0, cx
    [0, 800, 450],    # 0, fy, cy
    [0, 0, 1],
], dtype=np.float32)

CAM_IMAGE_SIZE = (1600, 900)  # (width, height)

# ==============================================================================
# STEROWANIE (Pure Pursuit z trajektorii modelu)
# ==============================================================================
# PID dla prędkości
SPEED_KP = 0.5
SPEED_KI = 0.1
SPEED_KD = 0.05
TARGET_SPEED_DEFAULT = 50 / 3.6  # 50 km/h → m/s
TARGET_SPEED_MAX = 100 / 3.6     # 100 km/h
TARGET_SPEED_MIN = 10 / 3.6      # 10 km/h

# PID dla skrętu
STEER_KP = 0.5
STEER_KI = 0.01
STEER_KD = 0.1

# Pure Pursuit - odległość lookahead (metry)
LOOKAHEAD_DIST = 5.0

# Hamowanie awaryjne - próg odległości (metry)
EMERGENCY_BRAKE_DIST = 3.0
EMERGENCY_BRAKE_FORCE = 1.0

# ==============================================================================
# ZMIANA PASA
# ==============================================================================
LANE_WIDTH = 3.5        # Szerokość pasa w metrach
LANE_CHANGE_DURATION = 2.0  # Czas zmiany pasa (sekundy)

# ==============================================================================
# NAWIGACJA - KOMENDY
# ==============================================================================
CMD_GO_STRAIGHT = 0
CMD_TURN_LEFT = 1
CMD_TURN_RIGHT = 2

CMD_NAMES = {
    CMD_GO_STRAIGHT: 'Go Straight',
    CMD_TURN_LEFT: 'Turn Left',
    CMD_TURN_RIGHT: 'Turn Right',
}

# One-hot encoding dla modelu SparseDrive
# [Turn Right, Turn Left, Go Straight]
CMD_TO_ONEHOT = {
    CMD_GO_STRAIGHT: [0, 0, 1],
    CMD_TURN_LEFT: [0, 1, 0],
    CMD_TURN_RIGHT: [1, 0, 0],
}

# Rozszerzone stany dla panelu sterowania
STATE_DRIVING_STRAIGHT = 'JAZDA PROSTO'
STATE_TURNING_LEFT = 'SKRET W LEWO'
STATE_TURNING_RIGHT = 'SKRET W PRAWO'
STATE_OVERTAKING_LEFT = 'WYPRZEDZANIE Z LEWEJ'
STATE_OVERTAKING_RIGHT = 'WYPRZEDZANIE Z PRAWEJ'
STATE_LANE_CHANGE_LEFT = 'ZMIANA PASA W LEWO'
STATE_LANE_CHANGE_RIGHT = 'ZMIANA PASA W PRAWO'
STATE_U_TURN = 'ZAWROT'
STATE_EMERGENCY_BRAKE = 'HAMOWANIE AWARYJNE'
STATE_STOPPED = 'ZATRZYMANY'

# ==============================================================================
# WIZUALIZACJA
# ==============================================================================
# BEV
BEV_XLIM = 40   # metry w każdą stronę
BEV_YLIM = 40
BEV_UPDATE_INTERVAL = 3  # aktualizuj BEV co N klatek (dla wydajności matplotlib)

# Widok kamer
CAMERA_TILE_SIZE = (400, 225)  # rozmiar jednej kafelki w widoku 3x2

# Panel sterowania
PANEL_WIDTH = 400
PANEL_HEIGHT = 700

# ==============================================================================
# MODEL SPARSEDRIVE
# ==============================================================================
SPARSEDRIVE_CONFIG = 'SparseDrive-main/projects/configs/sparsedrive_small_stage2.py'
SPARSEDRIVE_CHECKPOINT = 'SparseDrive-main/ckpt/sparsedrive_stage2.pth'
RESNET_CHECKPOINT = 'SparseDrive-main/ckpt/resnet50-19c8e357.pth'
INPUT_SHAPE = (704, 256)  # (width, height) - model input
IMG_NORM_MEAN = [123.675, 116.28, 103.53]
IMG_NORM_STD = [58.395, 57.12, 57.375]
QUEUE_LENGTH = 4
NUM_CAMS = 6

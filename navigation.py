"""System nawigacji i komend dla autonomicznej jazdy.

Mapuje przyciski panelu sterowania na komendy zrozumiałe dla modelu SparseDrive.
Obsługuje: jazdę prosto, skręty, zmianę pasa, zawracanie.
"""
import time
import math
from config import (
    CMD_GO_STRAIGHT, CMD_TURN_LEFT, CMD_TURN_RIGHT,
    CMD_TO_ONEHOT, CMD_NAMES,
    STATE_DRIVING_STRAIGHT, STATE_TURNING_LEFT, STATE_TURNING_RIGHT,
    STATE_OVERTAKING_LEFT, STATE_OVERTAKING_RIGHT,
    STATE_LANE_CHANGE_LEFT, STATE_LANE_CHANGE_RIGHT,
    STATE_U_TURN, STATE_EMERGENCY_BRAKE,
    LANE_WIDTH, LANE_CHANGE_DURATION,
)


class CommandSystem:
    """Zarządza komendami nawigacyjnymi i stanem jazdy."""

    def __init__(self):
        self._current_cmd = CMD_GO_STRAIGHT
        self._pending_cmd = None
        self._onehot = CMD_TO_ONEHOT[CMD_GO_STRAIGHT]

        # Stan zmiany pasa
        self._lane_change_active = False
        self._lane_change_direction = 0  # -1 = lewo, +1 = prawo
        self._lane_change_start_time = 0
        self._lane_change_progress = 0.0

        # Stan zawracania
        self._uturn_active = False
        self._uturn_stage = 0  # 0 = pierwszy skręt, 1 = drugi skręt, 2 = zakończony
        self._uturn_start_yaw = 0

        # Stan hamowania
        self._emergency_brake_active = False

        # Aktualny status do wyświetlenia
        self.status = STATE_DRIVING_STRAIGHT
        self.status_message = ''

    @property
    def current_command(self):
        return self._current_cmd

    @property
    def onehot(self):
        return self._onehot

    @property
    def is_lane_changing(self):
        return self._lane_change_active

    @property
    def lane_change_offset(self):
        """Zwraca lateralny offset dla zmiany pasa w metrach."""
        if not self._lane_change_active:
            return 0.0
        elapsed = time.time() - self._lane_change_start_time
        progress = min(1.0, elapsed / LANE_CHANGE_DURATION)
        self._lane_change_progress = progress

        # Płynna interpolacja sinusem (ease-in-out)
        smooth = math.sin(progress * math.pi / 2) ** 2
        offset = smooth * LANE_WIDTH * self._lane_change_direction

        if progress >= 1.0:
            self._lane_change_active = False
            self._lane_change_progress = 0.0
            self._lane_change_direction = 0
            self._update_status()

        return offset

    def set_command(self, cmd: int):
        """Ustawia komendę nawigacyjną."""
        if cmd in (CMD_GO_STRAIGHT, CMD_TURN_LEFT, CMD_TURN_RIGHT):
            self._current_cmd = cmd
            self._onehot = CMD_TO_ONEHOT[cmd]
            self._uturn_active = False
            self._uturn_stage = 0
            self._update_status()

    def start_lane_change(self, direction: int):
        """Rozpoczyna zmianę pasa. direction: -1 (lewo) lub +1 (prawo)."""
        self._lane_change_active = True
        self._lane_change_direction = direction
        self._lane_change_start_time = time.time()
        self._lane_change_progress = 0.0
        if direction == -1:
            self.status = STATE_LANE_CHANGE_LEFT
        else:
            self.status = STATE_LANE_CHANGE_RIGHT

    def start_uturn(self, current_yaw: float):
        """Rozpoczyna sekwencję zawracania."""
        self._uturn_active = True
        self._uturn_stage = 0
        self._uturn_start_yaw = current_yaw
        self._current_cmd = CMD_TURN_LEFT
        self._onehot = CMD_TO_ONEHOT[CMD_TURN_LEFT]
        self.status = STATE_U_TURN

    def update_uturn(self, current_yaw: float):
        """Aktualizuje stan zawracania na podstawie kąta odchylenia."""
        if not self._uturn_active:
            return

        yaw_diff = abs(current_yaw - self._uturn_start_yaw)
        yaw_diff = min(yaw_diff, 2 * math.pi - yaw_diff)

        if self._uturn_stage == 0 and yaw_diff > math.radians(150):
            # Pierwszy skręt prawie zakończony, drugi skręt
            self._uturn_stage = 1
            self._current_cmd = CMD_TURN_LEFT
            self._onehot = CMD_TO_ONEHOT[CMD_TURN_LEFT]
        elif self._uturn_stage == 1 and yaw_diff < math.radians(20):
            # Wróciliśmy do jazdy prosto (zawrót zakończony)
            self._uturn_stage = 2
            self._uturn_active = False
            self._current_cmd = CMD_GO_STRAIGHT
            self._onehot = CMD_TO_ONEHOT[CMD_GO_STRAIGHT]
            self._update_status()

    def set_emergency_brake(self, active: bool):
        """Aktywuje/dezaktywuje hamowanie awaryjne."""
        self._emergency_brake_active = active
        if active:
            self.status = STATE_EMERGENCY_BRAKE
        else:
            self._update_status()

    @property
    def emergency_brake(self):
        return self._emergency_brake_active

    def _update_status(self):
        """Aktualizuje tekst statusu na podstawie bieżącego stanu."""
        if self._emergency_brake_active:
            self.status = STATE_EMERGENCY_BRAKE
        elif self._uturn_active:
            self.status = STATE_U_TURN
        elif self._lane_change_active:
            if self._lane_change_direction == -1:
                self.status = STATE_LANE_CHANGE_LEFT
            else:
                self.status = STATE_LANE_CHANGE_RIGHT
        elif self._current_cmd == CMD_TURN_LEFT:
            self.status = STATE_TURNING_LEFT
        elif self._current_cmd == CMD_TURN_RIGHT:
            self.status = STATE_TURNING_RIGHT
        else:
            self.status = STATE_DRIVING_STRAIGHT

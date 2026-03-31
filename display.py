"""
display.py — Display / Overlay Module
======================================
Renders all on-screen text overlays onto the video frame:
  - Gesture label above each detected hand
  - Handedness label (Left / Right)
  - Hand count in the top-left corner
  - FPS counter
  - Key hints at the bottom of the screen

Keeping rendering logic here keeps main.py clean and focused
on the control flow rather than UI details.

Usage:
    from display import Overlay
    overlay = Overlay()
    overlay.draw_gesture_info(frame, gesture_results)
    overlay.draw_hand_count(frame, hand_count=2)
    overlay.draw_fps(frame)
    overlay.draw_instructions(frame)
"""

import cv2
import time


# ── Color palette (BGR format for OpenCV) ──────────────────────────────────
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_GREEN  = (0, 220, 0)
COLOR_CYAN   = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED    = (0, 0, 220)
COLOR_YELLOW = (0, 255, 255)

# Map handedness to a color for the label
HAND_COLORS = {
    "Right": COLOR_GREEN,
    "Left":  COLOR_CYAN,
    "Unknown": COLOR_WHITE,
}


class Overlay:
    """
    Draws informational overlays onto OpenCV frames.
    """

    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX):
        """
        Initialize overlay settings.

        Args:
            font: OpenCV font constant (default: FONT_HERSHEY_SIMPLEX).
        """
        self.font = font
        self._fps_timer = time.time()
        self._frame_count = 0
        self._fps = 0.0

    # ── Main drawing method ────────────────────────────────────────────────

    def draw_gesture_info(self, frame, gesture_results: list[dict]):
        """
        Draw gesture label and handedness text above each detected hand.

        For each hand, draws:
          1. The handedness label ("Right" / "Left") in a colored box.
          2. The recognized gesture name directly above the hand's bounding box.
          3. A finger-state bar showing which fingers are extended.

        Args:
            frame:           BGR image to annotate (modified in-place).
            gesture_results: Output of GestureRecognizer.recognize_all().
        """
        for result in gesture_results:
            bb         = result["bounding_box"]
            gesture    = result["gesture"]
            handedness = result["handedness"]
            finger_states = result.get("finger_states", [])

            color = HAND_COLORS.get(handedness, COLOR_WHITE)

            # ── Gesture label ───────────────────────────────────────────────
            label_y = max(bb["y"] - 50, 30)
            label_x = bb["x"]

            gesture_text = f"{gesture}"
            (text_w, text_h), _ = cv2.getTextSize(gesture_text, self.font, 1.0, 2)
            cv2.rectangle(
                frame,
                (label_x - 5, label_y - text_h - 10),
                (label_x + text_w + 5, label_y + 5),
                COLOR_BLACK,
                cv2.FILLED,
            )
            cv2.putText(
                frame, gesture_text,
                (label_x, label_y),
                self.font, 1.0, color, 2, cv2.LINE_AA,
            )

            # ── Handedness label ────────────────────────────────────────────
            hand_text = handedness
            (hw, hh), _ = cv2.getTextSize(hand_text, self.font, 0.7, 2)
            hand_y = label_y - text_h - 20
            cv2.rectangle(
                frame,
                (label_x - 5, hand_y - hh - 8),
                (label_x + hw + 5, hand_y + 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                frame, hand_text,
                (label_x, hand_y),
                self.font, 0.7, COLOR_BLACK, 2, cv2.LINE_AA,
            )

            # ── Finger state bar ────────────────────────────────────────────
            if finger_states:
                self._draw_finger_bar(frame, finger_states, bb, color)

    def _draw_finger_bar(self, frame, states: list[int], bb: dict, color):
        """
        Draw a small row of 5 circles below the hand.
        Filled circle = finger up; empty circle = finger down.
        """
        finger_labels = ["T", "I", "M", "R", "P"]
        bar_y   = bb["y"] + bb["h"] + 25
        radius  = 12
        spacing = 30
        start_x = bb["x"] + 5

        for i, (state, label) in enumerate(zip(states, finger_labels)):
            cx = start_x + i * spacing
            if state == 1:
                cv2.circle(frame, (cx, bar_y), radius, color, cv2.FILLED)
                cv2.putText(frame, label, (cx - 6, bar_y + 5),
                            self.font, 0.4, COLOR_BLACK, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, bar_y), radius, color, 2)
                cv2.putText(frame, label, (cx - 6, bar_y + 5),
                            self.font, 0.4, color, 1, cv2.LINE_AA)

    # ── Status panel ──────────────────────────────────────────────────────

    def draw_hand_count(self, frame, hand_count: int):
        """Display the number of detected hands in the top-left corner."""
        text = f"Hands: {hand_count}"
        cv2.putText(
            frame, text,
            (10, 35),
            self.font, 1.0,
            COLOR_YELLOW if hand_count > 0 else COLOR_WHITE,
            2, cv2.LINE_AA,
        )

    def draw_fps(self, frame):
        """Compute and display the current FPS in the top-right corner."""
        self._frame_count += 1
        elapsed = time.time() - self._fps_timer

        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()

        fps_text = f"FPS: {self._fps:.1f}"
        h, w, _ = frame.shape
        (tw, _), _ = cv2.getTextSize(fps_text, self.font, 0.8, 2)
        cv2.putText(
            frame, fps_text,
            (w - tw - 10, 35),
            self.font, 0.8, COLOR_ORANGE, 2, cv2.LINE_AA,
        )

    def draw_instructions(self, frame, logging_enabled: bool = False, saving_enabled: bool = False):
        """Draw key hint bar at the bottom of the frame."""
        h, w, _ = frame.shape
        hints = [
            "Q: Quit",
            "S: Save frame" if saving_enabled else "",
            f"L: Log {'ON' if logging_enabled else 'OFF'}" if saving_enabled else "",
        ]
        hint_text = "  |  ".join(h for h in hints if h)
        (tw, th), _ = cv2.getTextSize(hint_text, self.font, 0.55, 1)

        cv2.rectangle(frame, (0, h - th - 18), (w, h), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, hint_text, (10, h - 8),
                    self.font, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    def draw_status_message(self, frame, message: str, color=None):
        """Draw a centered status message (e.g., 'Frame Saved!')."""
        if color is None:
            color = COLOR_YELLOW
        h, w, _ = frame.shape
        (tw, th), _ = cv2.getTextSize(message, self.font, 1.2, 2)
        cx = (w - tw) // 2
        cy = h // 2
        cv2.putText(frame, message, (cx, cy),
                    self.font, 1.2, color, 3, cv2.LINE_AA)
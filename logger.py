"""
logger.py — Gesture Logger Module (Optional)
=============================================
Records detected gestures with timestamps to a CSV log file.
Also saves frame snapshots when S is pressed.

Log file format:
    timestamp, handedness, gesture, finger_states
    2024-01-15 12:00:01.234, Right, Peace, [0,1,1,0,0]

Usage:
    from logger import GestureLogger
    gl = GestureLogger(log_file="gestures.csv", save_dir="frames")
    gl.log(gesture_results)
    gl.save_frame(frame, "Peace")
    gl.close()
"""

import csv
import os
import cv2
from datetime import datetime


class GestureLogger:
    """
    Records gesture detections to a CSV file and saves frame images.
    """

    def __init__(self, log_file: str = "gestures.csv", save_dir: str = "saved_frames"):
        """
        Set up the logger — create CSV file and save directory.

        Args:
            log_file: Filename for the gesture log (CSV format).
            save_dir: Directory to save frame snapshots.
        """
        self.log_file = log_file
        self.save_dir = save_dir

        # Create the frames directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Open CSV and write header row
        self._csv_file = open(log_file, mode="w", newline="", buffering=1)
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["timestamp", "handedness", "gesture", "finger_states"]
        )
        self._writer.writeheader()

        print(f"[GestureLogger] Logging to '{log_file}'. Frames saved to '{save_dir}/'.")

    def log(self, gesture_results: list[dict]):
        """
        Write gesture detection results to the CSV.

        Args:
            gesture_results: List of gesture dicts from GestureRecognizer.recognize_all().
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        for result in gesture_results:
            self._writer.writerow({
                "timestamp":    now,
                "handedness":   result.get("handedness", "Unknown"),
                "gesture":      result.get("gesture",    "Unknown"),
                "finger_states": str(result.get("finger_states", [])),
            })

    def save_frame(self, frame, label: str = "frame"):
        """
        Save the current frame as a JPEG image with gesture label in the filename.

        Args:
            frame: NumPy array (BGR image) to save.
            label: Gesture label used in the filename.
        """
        safe_label = label.replace(" ", "_")
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename   = os.path.join(self.save_dir, f"{safe_label}_{timestamp}.jpg")

        cv2.imwrite(filename, frame)
        print(f"[GestureLogger] Frame saved: {filename}")

    def close(self):
        """Flush and close the CSV file."""
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()
            print(f"[GestureLogger] Log file closed: '{self.log_file}'.")
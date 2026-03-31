import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

LANDMARK_NAMES = {
    0: "WRIST", 1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP", 6: "INDEX_FINGER_PIP", 7: "INDEX_FINGER_DIP", 8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP", 10: "MIDDLE_FINGER_PIP", 11: "MIDDLE_FINGER_DIP", 12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP", 14: "RING_FINGER_PIP", 15: "RING_FINGER_DIP", 16: "RING_FINGER_TIP",
    17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
}

FINGER_TIPS = [4, 8, 12, 16, 20]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


class HandDetector:

    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        if not os.path.exists(MODEL_PATH):
            print("[HandDetector] Downloading hand landmark model (one time only)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[HandDetector] Model downloaded successfully.")

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        print(f"[HandDetector] Ready. max_hands={max_hands}")

    def detect(self, frame_rgb):
        frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)
        return results

    def draw(self, frame_bgr, results):
        if not results.hand_landmarks:
            return frame_bgr
        h, w, _ = frame_bgr.shape
        for hand_landmarks in results.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame_bgr, points[start], points[end], (0, 180, 0), 2)
            for px, py in points:
                cv2.circle(frame_bgr, (px, py), 5, (255, 255, 255), -1)
                cv2.circle(frame_bgr, (px, py), 5, (0, 200, 0), 1)
        return frame_bgr

    def get_hands_data(self, results, frame_shape):
        height, width, _ = frame_shape
        hands_data = []
        if not results.hand_landmarks:
            return hands_data
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = "Unknown"
            if results.handedness and idx < len(results.handedness):
                handedness = results.handedness[idx][0].category_name
            landmarks = []
            x_coords, y_coords = [], []
            for lm_id, lm in enumerate(hand_landmarks):
                px = int(lm.x * width)
                py = int(lm.y * height)
                x_coords.append(px)
                y_coords.append(py)
                landmarks.append({"id": lm_id, "name": LANDMARK_NAMES.get(lm_id, f"LM_{lm_id}"), "x": px, "y": py, "z": lm.z})
            padding = 20
            bounding_box = {
                "x": max(0, min(x_coords) - padding),
                "y": max(0, min(y_coords) - padding),
                "w": min(width, max(x_coords) + padding) - max(0, min(x_coords) - padding),
                "h": min(height, max(y_coords) + padding) - max(0, min(y_coords) - padding),
            }
            hands_data.append({"handedness": handedness, "landmarks": landmarks, "bounding_box": bounding_box})
        return hands_data

    def count_hands(self, results):
        return len(results.hand_landmarks) if results.hand_landmarks else 0

    def close(self):
        self.detector
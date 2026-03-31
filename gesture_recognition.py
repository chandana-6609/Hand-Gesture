"""
gesture_recognition.py — Gesture Recognition Module
=====================================================
Classifies hand poses into named gestures by analyzing the positions
of MediaPipe hand landmarks.

How it works:
  - Each hand has 21 landmarks. We focus on the 5 fingertip-to-base
    relationships to determine whether each finger is "up" (extended)
    or "down" (curled).
  - A "finger state" array like [0, 1, 1, 0, 0] means:
      Thumb=down, Index=up, Middle=up, Ring=down, Pinky=down → Peace sign
  - We compare the detected state to a lookup table of known gestures.
  - For the thumb, we check X-axis distance (left/right) because the thumb
    folds horizontally, not vertically like the other fingers.

Supported gestures:
  - Open Palm    → all 5 fingers extended
  - Fist         → all fingers curled
  - Thumbs Up    → only thumb extended
  - Peace        → index + middle extended
  - Pointing     → only index finger extended
  - Three        → index + middle + ring extended
  - Rock On      → index + pinky extended
  - Gun          → thumb + index extended
  - Call Me      → thumb + pinky extended
  - Four         → index + middle + ring + pinky extended

Usage:
    from gesture_recognition import GestureRecognizer
    recognizer = GestureRecognizer()
    gesture_name = recognizer.recognize(hand_data)
"""

from hand_detection import FINGER_TIPS


# Map finger index → MCP (knuckle) landmark index
# MCP = Metacarpophalangeal joint (base knuckle of each finger)
FINGER_MCP = {
    0: 2,   # Thumb: compare tip (4) to IP joint (3)
    1: 5,   # Index finger MCP
    2: 9,   # Middle finger MCP
    3: 13,  # Ring finger MCP
    4: 17,  # Pinky MCP
}

# Gesture definitions: map a finger state tuple to a gesture label.
# State format: (Thumb, Index, Middle, Ring, Pinky) — 1=up/extended, 0=down/curled
GESTURE_MAP = {
    (1, 1, 1, 1, 1): "Open Palm",
    (0, 0, 0, 0, 0): "Fist",
    (1, 0, 0, 0, 0): "Thumbs Up",
    (0, 1, 0, 0, 0): "Pointing",
    (0, 1, 1, 0, 0): "Peace",
    (0, 1, 1, 1, 0): "Three",
    (0, 1, 1, 1, 1): "Four",
    (0, 1, 0, 0, 1): "Rock On",
    (1, 1, 0, 0, 0): "Gun",
    (1, 0, 0, 0, 1): "Call Me",
    (1, 1, 1, 0, 0): "Three (with thumb)",
}


class GestureRecognizer:
    """
    Classifies hand gestures from MediaPipe landmark data.

    The recognizer is purely geometric — no machine learning model is needed.
    It works by checking whether each finger tip is above its base knuckle
    in the image (y-coordinate decreases upward in image space).
    """

    def __init__(self):
        """Initialize with the built-in gesture map."""
        self.gesture_map = GESTURE_MAP
        print(f"[GestureRecognizer] Loaded {len(self.gesture_map)} gestures.")

    def get_finger_states(self, landmarks: list[dict], handedness: str) -> list[int]:
        """
        Determine which fingers are extended (up) vs. curled (down).

        For each finger:
          - Fingers 1–4 (Index → Pinky): if the fingertip's Y pixel is ABOVE
            (smaller Y value) its MCP knuckle, the finger is up.
          - Thumb (finger 0): uses X-axis because it folds sideways. The comparison
            direction depends on whether it's a Left or Right hand.

        Args:
            landmarks:  List of 21 landmark dicts from HandDetector.get_hands_data().
            handedness: "Left" or "Right" — affects thumb direction logic.

        Returns:
            List of 5 integers [Thumb, Index, Middle, Ring, Pinky] — 1=up, 0=down.
        """
        states = []

        for finger_idx in range(5):
            tip_id = FINGER_TIPS[finger_idx]        # Tip landmark index
            mcp_id = FINGER_MCP[finger_idx]         # Base knuckle landmark index

            tip = landmarks[tip_id]
            mcp = landmarks[mcp_id]

            if finger_idx == 0:
                # ── Thumb logic ──────────────────────────────────────────────
                # The thumb extends sideways. For a Right hand (mirrored view),
                # the tip should be to the LEFT of the MCP (smaller X).
                # For a Left hand it should be to the RIGHT.
                # We compare against landmark 3 (THUMB_IP) for accuracy.
                ip = landmarks[3]   # Thumb IP joint (between MCP and tip)
                if handedness == "Right":
                    # In mirrored/selfie view, right hand thumb extends left
                    states.append(1 if tip["x"] < ip["x"] else 0)
                else:
                    # Left hand thumb extends right
                    states.append(1 if tip["x"] > ip["x"] else 0)
            else:
                # ── Other fingers ─────────────────────────────────────────────
                # In image coordinates, Y=0 is at the TOP. So a smaller Y means
                # the point is HIGHER on the screen. If the tip's Y is LESS than
                # the MCP's Y, the finger is pointing upward → extended.
                states.append(1 if tip["y"] < mcp["y"] else 0)

        return states

    def recognize(self, hand_data: dict) -> str:
        """
        Recognize a gesture from a single hand's landmark data.

        Args:
            hand_data: A dict from HandDetector.get_hands_data() with keys:
                       'handedness', 'landmarks', 'bounding_box'.

        Returns:
            A string gesture label (e.g., "Open Palm", "Fist", "Peace").
            Returns "Unknown" if no match is found.
        """
        landmarks  = hand_data["landmarks"]
        handedness = hand_data["handedness"]

        # Step 1: Compute finger states (which fingers are up)
        states = self.get_finger_states(landmarks, handedness)
        state_tuple = tuple(states)

        # Step 2: Look up the gesture in our map
        gesture = self.gesture_map.get(state_tuple, "Unknown")

        return gesture

    def recognize_all(self, hands_data: list[dict]) -> list[dict]:
        """
        Recognize gestures for all detected hands simultaneously.

        Args:
            hands_data: List of hand dicts from HandDetector.get_hands_data().

        Returns:
            List of dicts with gesture info added:
              {
                "handedness": str,
                "gesture":    str,
                "finger_states": [int, int, int, int, int],
                "bounding_box": dict,
                "landmarks":    list,
              }
        """
        results = []
        for hand_data in hands_data:
            landmarks  = hand_data["landmarks"]
            handedness = hand_data["handedness"]

            finger_states = self.get_finger_states(landmarks, handedness)
            gesture = self.gesture_map.get(tuple(finger_states), "Unknown")

            results.append({
                "handedness":    handedness,
                "gesture":       gesture,
                "finger_states": finger_states,
                "bounding_box":  hand_data["bounding_box"],
                "landmarks":     landmarks,
            })

        return results

    def list_gestures(self) -> list[str]:
        """Return all gesture labels known to this recognizer."""
        return list(set(self.gesture_map.values()))
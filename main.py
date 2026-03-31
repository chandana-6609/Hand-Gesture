import cv2
import sys
import time

from camera import Camera
from hand_detection import HandDetector
from gesture_recognition import GestureRecognizer
from display import Overlay
from logger import GestureLogger

CAMERA_INDEX         = 0
MAX_HANDS            = 2
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE  = 0.5
ENABLE_LOGGING       = True
ENABLE_FRAME_SAVING  = True
LOG_FILE             = "gestures.csv"
SAVE_DIR             = "saved_frames"
STATUS_MSG_DURATION  = 1.5


def run():
    print("=" * 55)
    print("  Hand Gesture Recognition System")
    print("  Press Q to quit | S to save frame | L to toggle log")
    print("=" * 55)

    try:
        cam = Camera(camera_index=CAMERA_INDEX)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    detector   = HandDetector(max_hands=MAX_HANDS, detection_confidence=DETECTION_CONFIDENCE, tracking_confidence=TRACKING_CONFIDENCE)
    recognizer = GestureRecognizer()
    overlay    = Overlay()

    gesture_logger = None
    logging_active = ENABLE_LOGGING
    if ENABLE_LOGGING:
        gesture_logger = GestureLogger(log_file=LOG_FILE, save_dir=SAVE_DIR)

    status_msg       = ""
    status_msg_until = 0.0

    print("\n[Main] Starting detection loop. Press Q to quit.\n")

    while cam.is_opened():
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame     = cam.flip(frame, flip_code=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = detector.detect(frame_rgb)

        detector.draw(frame, results)

        hands_data      = detector.get_hands_data(results, frame.shape)
        gesture_results = recognizer.recognize_all(hands_data)

        overlay.draw_gesture_info(frame, gesture_results)
        overlay.draw_hand_count(frame, hand_count=len(hands_data))
        overlay.draw_fps(frame)
        overlay.draw_instructions(frame, logging_enabled=logging_active, saving_enabled=ENABLE_FRAME_SAVING)

        if logging_active and gesture_logger and gesture_results:
            gesture_logger.log(gesture_results)

        if time.time() < status_msg_until:
            overlay.draw_status_message(frame, status_msg)

        cam.show(frame)
        key = cam.wait_key(1)

        if key == ord("q") or key == ord("Q"):
            break

        if key == ord("s") or key == ord("S"):
            if ENABLE_FRAME_SAVING and gesture_logger:
                label = gesture_results[0]["gesture"] if gesture_results else "frame"
                gesture_logger.save_frame(frame, label)
                status_msg       = "Frame Saved!"
                status_msg_until = time.time() + STATUS_MSG_DURATION

        if key == ord("l") or key == ord("L"):
            if ENABLE_LOGGING:
                logging_active   = not logging_active
                status_msg       = f"Logging {'ON' if logging_active else 'OFF'}"
                status_msg_until = time.time() + STATUS_MSG_DURATION

    cam.release()
    detector.close()
    if gesture_logger:
        gesture_logger.close()


if __name__ == "__main__":
    run()

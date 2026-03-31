import cv2
import time

try:
    from vidgear.gears import CamGear
    VIDGEAR_AVAILABLE = True
except ImportError:
    VIDGEAR_AVAILABLE = False


class Camera:

    def __init__(self, camera_index=0, window_name="Hand Gesture Recognition"):
        self.camera_index = camera_index
        self.window_name = window_name
        self._use_vidgear = False
        self.stream = None
        self.cap = None

        if VIDGEAR_AVAILABLE:
            try:
                print(f"[Camera] Opening camera {camera_index} via VidGear...")
                self.stream = CamGear(source=camera_index, logging=False).start()
                time.sleep(1)
                test = self.stream.read()
                if test is not None and test.mean() > 3:
                    self._use_vidgear = True
                    print("[Camera] Camera ready (VidGear).")
                    return
                else:
                    self.stream.stop()
                    self.stream = None
            except Exception as e:
                print(f"[Camera] VidGear failed: {e}")
                self.stream = None

        # Fallback to OpenCV
        print(f"[Camera] Trying OpenCV fallback...")
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
            self.cap = cv2.VideoCapture(camera_index, backend) if backend else cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                break

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}.")

        for _ in range(30):
            self.cap.read()
            time.sleep(0.05)

        print("[Camera] Camera ready (OpenCV).")

    def read(self):
        if self._use_vidgear and self.stream:
            frame = self.stream.read()
            if frame is not None:
                return True, frame
            return False, None
        if self.cap:
            return self.cap.read()
        return False, None

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def flip(self, frame, flip_code=1):
        return cv2.flip(frame, flip_code)

    def wait_key(self, delay=1):
        return cv2.waitKey(delay) & 0xFF

    def release(self):
        if self.stream:
            self.stream.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[Camera] Camera released.")

    def is_opened(self):
        return True
import cv2

backends = [
    ("Default", None),
    ("DSHOW",   cv2.CAP_DSHOW),
    ("MSMF",    cv2.CAP_MSMF),
]

print("Testing cameras...\n")
found = []

for index in range(4):
    for name, backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if not cap.isOpened():
                continue
            for _ in range(5):
                cap.read()
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                continue
            mean = frame.mean()
            print(f"  Index {index} [{name}]: brightness={mean:.1f}", end="")
            if mean < 5:
                print("  <- BLACK (skip)")
                cap.release()
                continue
            print("  <- WORKS!")
            found.append((index, name))
            cv2.imshow(f"Camera {index} [{name}]", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            cap.release()
        except Exception as e:
            print(f"  Index {index} [{name}]: ERROR - {e}")

print("\n=== RESULT ===")
if found:
    for i, n in found:
        print(f"  CAMERA_INDEX={i}  backend={n}")
else:
    print("  No working camera found!")
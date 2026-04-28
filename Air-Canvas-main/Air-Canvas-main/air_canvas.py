import sys
# Prevent MediaPipe from triggering global TensorFlow conflicts
sys.modules['tensorflow'] = None

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time
import os
import subprocess
import threading

def setup_hardware():
    print("Scanning system for hardware accelerators...")
    try:
        output = subprocess.check_output(
            ["powershell", "-command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        gpus = [line.strip().upper() for line in output.split('\n') if line.strip()]
    except Exception:
        gpus = []

    preferred_device = ""
    active_gpu_name = ""
    
    # 1. First priority: NVIDIA or AMD Dedicated
    for g in gpus:
        if "NVIDIA" in g:
            preferred_device = "NVIDIA:GPU:0"
            active_gpu_name = g
            break
        elif "AMD" in g and "RADEON" in g and "GRAPHICS" not in g:
            preferred_device = "AMD:GPU:0" 
            active_gpu_name = g
            break
            
    # 2. Second priority: Integrated GPUs
    if not preferred_device:
        for g in gpus:
            if "INTEL" in g:
                preferred_device = "Intel:GPU:0"
                active_gpu_name = g
                break
            elif "AMD" in g:
                preferred_device = "AMD:GPU:0"
                active_gpu_name = g
                break
    
    if preferred_device:
        os.environ["OPENCV_OPENCL_DEVICE"] = preferred_device

    # Execute binding
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.useOpenCL():
            print(f"\nSUCCESS! Hardware Pipeline ACTIVE.")
            print(f"Using Hardware: GPU -> {active_gpu_name}")
            return
            
    # 3. Third priority: CPU Fallback
    cv2.ocl.setUseOpenCL(False)
    print("\nGPU binding failed. Hardware Pipeline FALLBACK.")
    print("Using Hardware: CPU -> Core CPU Processing")

setup_hardware()

# ================= CONFIG =================
BRUSH_THICKNESS = 3
ERASER_RADIUS = 30
PINCH_THRESHOLD = 40

DRAW_DELAY = 0.15
PINCH_STABLE_FRAMES = 2

prev_mid_x, prev_mid_y = 0, 0

# ================= CAMERA =================
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

print("\nBooting Asynchronous Camera Thread...")
cap = WebcamStream(0).start()

WIDTH = int(cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.stream.get(cv2.CAP_PROP_FPS)

print(f"SUCCESS! Camera Live Pipeline: {WIDTH}x{HEIGHT} @ {FPS} FPS")

# ================= MEDIAPIPE TASKS API =================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.7
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Hand connection pairs for drawing landmarks
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

xp, yp = 0, 0
prev_x, prev_y = 0, 0

dragging = False
prev_drag_x, prev_drag_y = 0, 0

draw_start_time = None
pinch_counter = 0

cv2.namedWindow("Air Canvas PRO UX", cv2.WINDOW_NORMAL)

frame_timestamp_ms = 0

# ================= HELPERS =================
def get_bezier_curve(p0, p1, p2, steps=15):
    """Generates perfect C1 continuous smooth quadratic bezier curve."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = int(((1 - t) ** 2) * p0[0] + 2 * (1 - t) * t * p1[0] + (t ** 2) * p2[0])
        y = int(((1 - t) ** 2) * p0[1] + 2 * (1 - t) * t * p1[1] + (t ** 2) * p2[1])
        pts.append([x, y])
    return np.array(pts, np.int32).reshape((-1, 1, 2))

def fingers_up(lm):
    return [
        lm[8][2] < lm[6][2],
        lm[12][2] < lm[10][2],
        lm[16][2] < lm[14][2],
        lm[20][2] < lm[18][2]
    ]

def dist(p1, p2):
    return math.hypot(p2[1] - p1[1], p2[2] - p1[2])

def draw_hand_landmarks(img, lmList):
    """Draw hand landmarks and connections on the image."""
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(lmList) and end_idx < len(lmList):
            pt1 = (lmList[start_idx][1], lmList[start_idx][2])
            pt2 = (lmList[end_idx][1], lmList[end_idx][2])
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    # Draw landmarks
    for _, x, y in lmList:
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

# ================= MAIN LOOP =================
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image from the frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    
    # Process with HandLandmarker (VIDEO mode requires timestamp)
    frame_timestamp_ms += 33  # ~30fps increment
    results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    lmList = []

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        
        h, w, _ = img.shape
        for id, lm in enumerate(hand_landmarks):
            lmList.append((id, int(lm.x * w), int(lm.y * h)))
        
        # Draw landmarks on the image
        draw_hand_landmarks(img, lmList)

    if lmList:
        fingers = fingers_up(lmList)

        x1, y1 = lmList[8][1], lmList[8][2]

        # ========= ADAPTIVE VELOCITY FILTER =========
        if prev_x == 0:
            prev_x, prev_y = x1, y1

        # Calculate exact pixel velocity
        speed = math.hypot(x1 - prev_x, y1 - prev_y)
        
        # 1Euro Style Scaling: 
        # Move fast -> Alpha approaches 1.0 (Zero smoothing, instant snapping)
        # Move slow -> Alpha approaches 0.1 (High smoothing, no jitter)
        alpha = min(1.0, 0.1 + (speed / 100.0))

        x1 = int(prev_x * (1 - alpha) + x1 * alpha)
        y1 = int(prev_y * (1 - alpha) + y1 * alpha)

        prev_x, prev_y = x1, y1

        # cursor
        cv2.circle(img, (x1, y1), 4, (0, 255, 255), -1)

        # ========= PINCH =========
        pinch_dist = dist(lmList[8], lmList[4])

        if pinch_dist < PINCH_THRESHOLD:
            pinch_counter += 1
        else:
            pinch_counter = 0

        pinch_active = pinch_counter > PINCH_STABLE_FRAMES

        # ========= GESTURE PRIORITY =========

        # ✊ ERASE (STRICT — NO MOVEMENT)
        if not any(fingers):
            dragging = False
            pinch_counter = 0
            draw_start_time = None

            cv2.circle(canvas, (x1, y1), ERASER_RADIUS, (0, 0, 0), -1)
            xp, yp = 0, 0

        # 🤏 MOVE
        elif pinch_active:
            draw_start_time = None

            if not dragging:
                dragging = True
                prev_drag_x, prev_drag_y = x1, y1

            dx = x1 - prev_drag_x
            dy = y1 - prev_drag_y

            M = np.float32([[1, 0, dx], [0, 1, dy]])
            canvas = cv2.warpAffine(canvas, M, (WIDTH, HEIGHT))

            prev_drag_x, prev_drag_y = x1, y1
            xp, yp = 0, 0

        # ☝️ DRAW
        elif fingers[0] and not any(fingers[1:]):

            if draw_start_time is None:
                draw_start_time = time.time()

            elif time.time() - draw_start_time > DRAW_DELAY:

                if xp == 0:
                    xp, yp = x1, y1
                    prev_mid_x, prev_mid_y = x1, y1
                else:
                    mid_x = (xp + x1) // 2
                    mid_y = (yp + y1) // 2

                    # Quadratic Bezier Smoothing (Constant Thickness + Round Caps)
                    curve_pts = get_bezier_curve((prev_mid_x, prev_mid_y), (xp, yp), (mid_x, mid_y), steps=20)
                    cv2.polylines(canvas, [curve_pts], False, (0, 0, 255), BRUSH_THICKNESS, cv2.LINE_AA)
                    
                    # Ensure perfectly smooth edges/caps exactly like real digital ink
                    for pt in curve_pts:
                        cv2.circle(canvas, tuple(pt[0]), BRUSH_THICKNESS // 2, (0, 0, 255), -1, cv2.LINE_AA)

                    xp, yp = x1, y1
                    prev_mid_x, prev_mid_y = mid_x, mid_y

        else:
            dragging = False
            xp, yp = 0, 0

    else:
        xp, yp = 0, 0
        dragging = False
        draw_start_time = None
        pinch_counter = 0

    # ========= MERGE =========
    # Ultra-fast Numpy boolean mask composites 10x faster than OpenCV matrix ops at 1080p
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 20
    img[mask] = canvas[mask]


    cv2.imshow("Air Canvas PRO UX", img)

    # ========= KEYS =========
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if key == ord('c'):  # FULL RESET
        canvas[:] = 0
        xp, yp = 0, 0
        prev_x, prev_y = 0, 0
        draw_start_time = None
        dragging = False
        pinch_counter = 0

hand_landmarker.close()
cap.stop()
cv2.destroyAllWindows()

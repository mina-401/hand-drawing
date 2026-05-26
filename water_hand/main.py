import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import threading
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ── 설정 ──────────────────────────────────────────────
CAM_W, CAM_H = 1024, 720
PANEL_W      = 220
WIDTH        = CAM_W + PANEL_W
HEIGHT       = CAM_H
MODEL_PATH   = "hand_landmarker.task"
SAVE_DIR     = "saves"

os.makedirs(SAVE_DIR, exist_ok=True)

COLORS = [
    (0,   120, 255),
    (80,   80, 255),
    (50,  200,  50),
    (200,  50, 200),
    (0,   220, 220),
    (255, 255, 255),
]
color_idx   = 0
brush_color = COLORS[color_idx]
brush_size  = 8

TIP = [4, 8, 12, 16, 20]
MCP = [2, 5,  9, 13, 17]

# ── 폰트 ──────────────────────────────────────────────
try:
    font_emoji = ImageFont.truetype("seguiemj.ttf", 28)
    font_title = ImageFont.truetype("arialbd.ttf",  15)
    font_sub   = ImageFont.truetype("arial.ttf",    12)
    font_small = ImageFont.truetype("arial.ttf",    11)
except:
    font_emoji = font_title = font_sub = font_small = ImageFont.load_default()

# ── MediaPipe (별도 스레드) ────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# 스레드 공유 변수
_latest_lms   = None       # 최신 랜드마크
_frame_to_det = None       # 감지할 프레임
_det_lock     = threading.Lock()
_stop_event   = threading.Event()

def detection_worker():
    """MediaPipe 감지 전용 스레드"""
    global _latest_lms, _frame_to_det
    while not _stop_event.is_set():
        with _det_lock:
            frame = _frame_to_det
        if frame is None:
            time.sleep(0.001)
            continue
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)
        with _det_lock:
            _latest_lms    = result.hand_landmarks
            _frame_to_det  = None  # 처리 완료

det_thread = threading.Thread(target=detection_worker, daemon=True)
det_thread.start()

# ── 캔버스 & 상태 ──────────────────────────────────────
canvas     = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
prev_point = None

last_color_time       = 0
last_save_time        = 0
gesture_display_until = 0

BUFFER_SIZE   = 8
CONFIRM_RATIO = 0.65
gesture_buffer = deque(maxlen=BUFFER_SIZE)

# ── 선 스무딩 (베지어) ────────────────────────────────
SMOOTH_WINDOW = 5          # 스무딩에 사용할 최근 점 개수
draw_point_buf = deque(maxlen=SMOOTH_WINDOW)

def bezier_point(p0, p1, p2, t):
    """2차 베지어 곡선 위의 점 계산"""
    x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    return int(x), int(y)

def draw_smooth_stroke(canvas, buf, color, size):
    """버퍼의 점들을 베지어 곡선으로 부드럽게 그리기"""
    pts = list(buf)
    if len(pts) < 3:
        if len(pts) == 2:
            cv2.line(canvas, pts[0], pts[1], color, size, cv2.LINE_AA)
        return
    # 중간점들을 제어점 삼아 Catmull-Rom 스타일로 연결
    for i in range(len(pts) - 2):
        p0 = pts[i]
        p1 = pts[i + 1]
        p2 = pts[i + 2]
        mid0 = ((p0[0]+p1[0])//2, (p0[1]+p1[1])//2)
        mid1 = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        steps = max(int(np.hypot(mid1[0]-mid0[0], mid1[1]-mid0[1])), 8)
        prev = mid0
        for s in range(1, steps + 1):
            t   = s / steps
            cur = bezier_point(mid0, p1, mid1, t)
            cv2.line(canvas, prev, cur, color, size, cv2.LINE_AA)
            prev = cur

# ── 제스처 감지 ───────────────────────────────────────
def fingers_up(lms):
    MARGIN = 0.03
    return [lms[TIP[i]].y < lms[MCP[i]].y - MARGIN for i in range(1, 5)]

def classify_raw(lms):
    index, middle, ring, pinky = fingers_up(lms)
    if not index and not middle and not ring and not pinky:
        return "pen_up"
    if index and not middle and not ring and not pinky:
        return "draw"
    if index and middle and not ring and not pinky:
        return "color"
    if index and middle and ring and pinky:
        return "save"
    return "none"

def stable_gesture():
    if not gesture_buffer:
        return "none"
    counts = {}
    for g in gesture_buffer:
        counts[g] = counts.get(g, 0) + 1
    best  = max(counts, key=counts.get)
    return best if counts[best] / len(gesture_buffer) >= CONFIRM_RATIO else "none"

# ── 저장 ──────────────────────────────────────────────
def save_frame(frame):
    filename = os.path.join(SAVE_DIR, f"capture_{int(time.time())}.png")
    cv2.imwrite(filename, frame)
    print(f"저장됨: {filename}")

# ── PIL 텍스트 ─────────────────────────────────────────
def put_pil_text(frame, text, x, y, font, color=(255, 255, 255)):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img     = Image.fromarray(rgb)
    draw    = ImageDraw.Draw(img)
    r, g, b = color[2], color[1], color[0]
    draw.text((x, y), text, font=font, embedded_color=True, fill=(r, g, b))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ── UI ────────────────────────────────────────────────
def draw_toolbar(frame):
    cv2.rectangle(frame, (0, 0), (CAM_W, 60), (25, 25, 25), -1)
    for i, c in enumerate(COLORS):
        x = 20 + i * 70
        cv2.rectangle(frame, (x, 10), (x+50, 50), c, -1)
        if i == color_idx:
            cv2.rectangle(frame, (x-3, 7), (x+53, 53), (255, 255, 255), 2)
    cv2.putText(frame, f"Brush: {brush_size}  (+/- key)",
                (500, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

def draw_gesture_label(frame, gesture):
    labels = {
        "draw":   ("Drawing",        (0, 220, 120)),
        "pen_up": ("Pen Up",         (120, 120, 120)),
        "color":  ("Color Changed!", (0, 200, 255)),
    }
    if gesture in labels:
        text, color = labels[gesture]
        cv2.putText(frame, text, (20, CAM_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

def draw_cursor(frame, pos, gesture):
    if pos is None:
        return
    x, y  = pos
    color = brush_color if gesture == "draw" else (100, 100, 100)
    cv2.circle(frame, (x, y), brush_size // 2 + 4, color, 2)
    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

def draw_panel(frame, gesture):
    px = CAM_W
    cv2.rectangle(frame, (px, 0), (WIDTH, HEIGHT), (20, 20, 20), -1)
    cv2.line(frame, (px, 0), (px, HEIGHT), (60, 60, 60), 1)
    cv2.putText(frame, "Gestures", (px+18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 1)
    cv2.line(frame, (px+12, 52), (px+PANEL_W-12, 52), (60, 60, 60), 1)

    gestures = [
        ("draw",   "DRAW",   "☝️", "Index finger", "Draw on canvas"),
        ("pen_up", "PEN UP", "✊", "Fist",         "Lift pen"),
        ("color",  "COLOR",  "✌️", "V sign",       "Change color"),
    ]

    for idx, (key, title, emoji, sub1, sub2) in enumerate(gestures):
        y_base    = 68 + idx * 118
        is_active = (gesture == key)
        if is_active:
            cv2.rectangle(frame, (px+8,  y_base-4),
                          (px+PANEL_W-8, y_base+102), (40, 40, 40), -1)
            cv2.rectangle(frame, (px+8,  y_base-4),
                          (px+PANEL_W-8, y_base+102), (80, 80, 80),  1)
        emoji_color = (0, 220, 150) if is_active else (160, 160, 160)
        put_pil_text(frame, emoji, px+14, y_base+6,  font_emoji, emoji_color)
        title_color = (255, 255, 255) if is_active else (180, 180, 180)
        put_pil_text(frame, title, px+58, y_base+10, font_title, title_color)
        put_pil_text(frame, sub1,  px+58, y_base+34, font_sub,   (140, 140, 140))
        put_pil_text(frame, sub2,  px+58, y_base+54, font_sub,   (100, 100, 100))
        cv2.line(frame, (px+12, y_base+106),
                 (px+PANEL_W-12, y_base+106), (40, 40, 40), 1)

    cv2.line(frame, (px+12, HEIGHT-82), (px+PANEL_W-12, HEIGHT-82), (50, 50, 50), 1)
    put_pil_text(frame, "Shortcuts",          px+16, HEIGHT-76, font_small, (120, 120, 120))
    put_pil_text(frame, "+/-  Brush size",    px+16, HEIGHT-58, font_small, (100, 100, 100))
    put_pil_text(frame, "S  Save",            px+16, HEIGHT-40, font_small, (100, 100, 100))
    put_pil_text(frame, "C  Clear  Q  Quit",  px+16, HEIGHT-22, font_small, (100, 100, 100))

# ── 메인 루프 ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

print("실행 중! S=저장 / C=초기화 / Q=종료")
prev_time = time.time()

while True:
    ret, cam_frame = cap.read()
    if not ret:
        break

    cam_frame = cv2.flip(cam_frame, 1)
    cam_frame = cv2.resize(cam_frame, (CAM_W, CAM_H))
    full      = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # 감지 스레드에 프레임 전달
    with _det_lock:
        if _frame_to_det is None:
            _frame_to_det = cam_frame.copy()
        lms_snapshot = _latest_lms

    gesture   = "none"
    index_pos = None

    if lms_snapshot:
        lms = lms_snapshot[0]
        gesture_buffer.append(classify_raw(lms))
        gesture = stable_gesture()

        lm_i      = lms[8]
        index_pos = (int(lm_i.x * CAM_W), int(lm_i.y * CAM_H))
        now       = time.time()

        if gesture == "draw" and index_pos and index_pos[1] > 60:
            draw_point_buf.append(index_pos)
            draw_smooth_stroke(canvas, draw_point_buf, brush_color, brush_size)

        elif gesture == "pen_up":
            draw_point_buf.clear()
            prev_point = None

        elif gesture == "color":
            draw_point_buf.clear()
            prev_point = None
            if now - last_color_time > 1.0:
                color_idx             = (color_idx + 1) % len(COLORS)
                brush_color           = COLORS[color_idx]
                last_color_time       = now
                gesture_display_until = now + 1.0

        else:
            draw_point_buf.clear()
            prev_point = None

    else:
        draw_point_buf.clear()
        gesture_buffer.clear()

    # 합성
    mask2d  = np.any(canvas > 0, axis=2)
    blended = cam_frame.copy()
    blended[mask2d] = canvas[mask2d]

    draw_toolbar(blended)
    draw_cursor(blended, index_pos, gesture)

    show = gesture if gesture != "none" or time.time() < gesture_display_until else "none"
    draw_gesture_label(blended, show)

    full[:, :CAM_W] = blended
    draw_panel(full, gesture)

    # FPS
    curr_time = time.time()
    fps       = 1.0 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time
    cv2.putText(full, f"FPS {fps:.0f}", (CAM_W-90, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow("Gesture Drawing", full)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
        draw_point_buf.clear()
    elif key == ord('s'):
        clean = cam_frame.copy()
        clean[mask2d] = canvas[mask2d]
        save_frame(clean)
    elif key in (ord('+'), ord('=')):
        brush_size = min(50, brush_size + 2)
    elif key == ord('-'):
        brush_size = max(2, brush_size - 2)

_stop_event.set()
cap.release()
cv2.destroyAllWindows()
detector.close()
print("종료")
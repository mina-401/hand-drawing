import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
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

# 손가락 랜드마크 인덱스
TIP = [4, 8, 12, 16, 20]
MCP = [2, 5,  9, 13, 17]  # 손가락 뿌리 (MCP 관절)

# ── 폰트 ──────────────────────────────────────────────
try:
    font_emoji = ImageFont.truetype("seguiemj.ttf", 28)
    font_title = ImageFont.truetype("arialbd.ttf",  15)
    font_sub   = ImageFont.truetype("arial.ttf",    12)
    font_small = ImageFont.truetype("arial.ttf",    11)
except:
    font_emoji = font_title = font_sub = font_small = ImageFont.load_default()

# ── MediaPipe ─────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

canvas     = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
prev_point = None

# ── 상태 변수 ─────────────────────────────────────────
last_color_time       = 0
last_save_time        = 0
gesture_display_until = 0

# 제스처 스무딩 버퍼
BUFFER_SIZE    = 8      # 최근 N프레임
CONFIRM_RATIO  = 0.65   # 65% 이상 동의해야 채택
gesture_buffer = deque(maxlen=BUFFER_SIZE)

# ── 제스처 감지 ───────────────────────────────────────
def fingers_up(lms):
    """
    MCP(뿌리) 기준으로 판단 — pip보다 엄격해서 오인식 감소
    tip.y < mcp.y - margin 이어야 '펴진' 것으로 인정
    """
    MARGIN = 0.03  # 이 값 높일수록 더 확실히 펴야 인식
    up = []
    for i in range(1, 5):  # 검지~새끼
        up.append(lms[TIP[i]].y < lms[MCP[i]].y - MARGIN)
    return up  # [검지, 중지, 약지, 새끼]

def classify_raw(lms):
    """단일 프레임 제스처 분류"""
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
    """버퍼 내 최다 제스처가 CONFIRM_RATIO 이상이면 반환, 아니면 none"""
    if not gesture_buffer:
        return "none"
    counts = {}
    for g in gesture_buffer:
        counts[g] = counts.get(g, 0) + 1
    best  = max(counts, key=counts.get)
    ratio = counts[best] / len(gesture_buffer)
    return best if ratio >= CONFIRM_RATIO else "none"

# ── 저장 ──────────────────────────────────────────────
def save_full_frame(frame):
    filename = os.path.join(SAVE_DIR, f"capture_{int(time.time())}.png")
    cv2.imwrite(filename, frame)
    print(f"저장됨: {filename}")

# ── PIL 텍스트 렌더링 ──────────────────────────────────
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

    cv2.line(frame, (px+12, HEIGHT-72), (px+PANEL_W-12, HEIGHT-72), (50, 50, 50), 1)
    put_pil_text(frame, "Shortcuts",          px+16, HEIGHT-66, font_small, (120, 120, 120))
    put_pil_text(frame, "+/-  Brush size",    px+16, HEIGHT-48, font_small, (100, 100, 100))
    put_pil_text(frame, "S  Save,  C  Clear,  Q  Quit", px+16, HEIGHT-30, font_small, (100, 100, 100))

# ── 메인 루프 ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

print("실행 중! C키=초기화 / Q키=종료")
prev_time = time.time()
do_save   = False

while True:
    ret, cam_frame = cap.read()
    if not ret:
        break

    cam_frame = cv2.flip(cam_frame, 1)
    cam_frame = cv2.resize(cam_frame, (CAM_W, CAM_H))
    full      = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    rgb      = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    gesture   = "none"
    index_pos = None
    do_save   = False

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]

        gesture_buffer.append(classify_raw(lms))
        gesture = stable_gesture()

        lm_i      = lms[8]
        index_pos = (int(lm_i.x * CAM_W), int(lm_i.y * CAM_H))
        now       = time.time()

        if gesture == "draw" and index_pos and index_pos[1] > 60:
            if prev_point:
                cv2.line(canvas, prev_point, index_pos, brush_color, brush_size)
            prev_point = index_pos

        elif gesture == "pen_up":
            prev_point = None

        elif gesture == "color":
            prev_point = None
            if now - last_color_time > 1.0:
                color_idx             = (color_idx + 1) % len(COLORS)
                brush_color           = COLORS[color_idx]
                last_color_time       = now
                gesture_display_until = now + 1.0

        else:
            prev_point = None

    else:
        prev_point = None
        gesture_buffer.clear()

    # 합성
    mask2d  = np.any(canvas > 0, axis=2)   # 채널 하나라도 값 있으면 True
    blended = cam_frame.copy()
    blended[mask2d] = canvas[mask2d]


    draw_toolbar(blended)
    draw_cursor(blended, index_pos, gesture)

    show = gesture if gesture != "none" or time.time() < gesture_display_until else "none"
    draw_gesture_label(blended, show)

    full[:, :CAM_W] = blended
    draw_panel(full, gesture)

    curr_time = time.time()
    fps       = 1.0 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time
    cv2.putText(full, f"FPS {fps:.0f}", (CAM_W-90, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    if do_save:
        # UI 없이 카메라 + 그림만 합성해서 저장
        clean = cam_frame.copy()
        clean[mask2d] = canvas[mask2d]
        save_full_frame(clean)

    cv2.imshow("Gesture Drawing", full)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
    elif key in (ord('+'), ord('=')):
        brush_size = min(50, brush_size + 2)
    elif key == ord('-'):
        brush_size = max(2, brush_size - 2)
    elif key == ord('s'):
        clean = cam_frame.copy()
        clean[mask2d] = canvas[mask2d]
        save_full_frame(clean)

cap.release()
cv2.destroyAllWindows()
detector.close()
print("종료")
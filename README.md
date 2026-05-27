# ✋ Hand Gesture Drawing

웹캠과 손 제스처로 그림을 그리는 AI 기반 인터랙티브 드로잉 앱
> 손을 카메라에 들이대고 검지로 그림을 그려보세요!

---

## 핵심 기능

### 1. 제스처 인식 그리기 — 검지 하나로 캔버스에 자유롭게 그리기
### 2. 베지어 곡선 선 스무딩 — 손떨림 없이 부드러운 선

검지 위치를 매 프레임 직선으로 이으면 손 떨림이 그대로 나타남.  
최근 **5개 점을 버퍼에 쌓아 베지어 곡선으로 연결**해 부드러운 선을 그림.

```python
SMOOTH_WINDOW  = 5
draw_point_buf = deque(maxlen=SMOOTH_WINDOW)

def bezier_point(p0, p1, p2, t):
    """세 점을 이용한 2차 베지어 곡선 위의 점 계산"""
    x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    return int(x), int(y)

def draw_smooth_stroke(canvas, buf, color, size):
    pts = list(buf)
    for i in range(len(pts) - 2):
        p0, p1, p2 = pts[i], pts[i+1], pts[i+2]

        mid0 = ((p0[0]+p1[0])//2, (p0[1]+p1[1])//2)  # p0~p1 중간점
        mid1 = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)  # p1~p2 중간점

        # mid0 → p1(제어점) → mid1 을 잇는 곡선
        steps = max(int(np.hypot(mid1[0]-mid0[0], mid1[1]-mid0[1])), 8)
        prev  = mid0
        for s in range(1, steps + 1):
            t   = s / steps
            cur = bezier_point(mid0, p1, mid1, t)
            cv2.line(canvas, prev, cur, color, size, cv2.LINE_AA)
            prev = cur
```

```
p1이 제어점 역할 → 꺾이지 않고 자연스럽게 휘어짐
```

### 3. 멀티스레드 MediaPipe FPS 안정화

MediaPipe 감지를 메인 루프에서 실행하면 측정 환경에 따라 다르지만 감지에 수십ms 소요
**감지 전용 스레드를 분리**해 렌더링과 감지가 동시에 실행되도록 개선.

```python
# 스레드 간 공유 변수
_latest_lms   = None  # 감지 스레드가 결과 저장
_frame_to_det = None  # 메인 루프가 프레임 전달
_det_lock     = threading.Lock()  # 동시 접근 방지

def detection_worker():
    """MediaPipe 감지 전용 스레드 — 백그라운드에서 계속 실행"""
    while not _stop_event.is_set():
        with _det_lock:
            frame = _frame_to_det
        if frame is None:
            time.sleep(0.001)  # 프레임 없으면 대기
            continue
        result = detector.detect(...)
        with _det_lock:
            _latest_lms   = result.hand_landmarks  # 결과 저장
            _frame_to_det = None                   # 처리 완료 표시

# 메인 루프 — 감지를 기다리지 않고 최신 결과만 읽음
with _det_lock:
    if _frame_to_det is None:
        _frame_to_det = cam_frame.copy()  # 프레임 전달
    lms_snapshot = _latest_lms           # 결과 읽기
```

```
기존 구조 (순차):
  카메라 읽기 → [MediaPipe 감지 10~20ms 대기] → 렌더링 → 출력

멀티스레드 구조 (병렬):
  메인:    카메라 읽기 → 결과 읽기 → 렌더링 → 출력
  스레드:              → MediaPipe 감지 →  결과 저장
```
### 4. 제스처 버퍼 스무딩 — 최근 8프레임 기반 안정적인 제스처 인식

매 프레임마다 제스처를 판단하면 손 떨림으로 인한 오인식이 잦아짐.  
최근 **8프레임을 버퍼에 쌓아서 65% 이상 동의한 제스처만 채택**하는 방식으로 안정화.

```python
BUFFER_SIZE   = 8
CONFIRM_RATIO = 0.65
gesture_buffer = deque(maxlen=BUFFER_SIZE)

def stable_gesture():
    counts = {}
    for g in gesture_buffer:
        counts[g] = counts.get(g, 0) + 1
    best  = max(counts, key=counts.get)
    return best if counts[best] / len(gesture_buffer) >= CONFIRM_RATIO else "none"
```

### 5. 거리 비율 기반 제스쳐 판별 — 손 기울기에 강인한 draw/pen_up 구분

단순히 손가락 y좌표만 비교하면 손이 기울어졌을 때 오인식이 발생.  
**검지 끝 ~ 손목 거리 / 손 전체 크기** 비율로 판별해 손 기울기에 강인하게 개선.

```python
def finger_extend_ratio(lms):
    tip   = lms[8]   # 검지 끝
    wrist = lms[0]   # 손목
    mid   = lms[9]   # 중지 뿌리 (손 크기 기준점)

    hand_size = np.hypot(wrist.x - mid.x, wrist.y - mid.y)  # 손 전체 크기
    extend    = np.hypot(tip.x - wrist.x, tip.y - wrist.y)  # 검지 뻗은 길이

    return extend / (hand_size + 1e-6)  # 비율 반환

def classify_raw(lms):
    ratio = finger_extend_ratio(lms)

    if ratio > 0.9:    # 검지가 확실히 뻗어있음
        return "draw"
    elif ratio < 0.5:  # 검지가 완전히 접혀있음
        return "pen_up"
    else:
        return "none"  # 애매한 전환 중 → 무시
```

```
손 기울기와 상관없이 비율은 일정하게 유지됨

검지 펼침:  extend / hand_size = 1.1 → draw
검지 접힘:  extend / hand_size = 0.4 → pen_up
전환 중:    extend / hand_size = 0.7 → none (무시)
```
---

## 🤚 제스처 조작

| 제스처 | 동작 |
|--------|------|
| ☝️ 검지만 펴기 | 그리기 |
| ✊ 주먹 쥐기 | 펜 올리기 |
| ✌️ 브이 | 색상 변경 |

---

## ⌨️ 단축키

| 키 | 동작 |
|----|------|
| `S` | 캡처 저장 (`saves/` 폴더) |
| `C` | 캔버스 초기화 |
| `+` / `-` | 브러시 크기 조절 |
| `Q` | 종료 |

---

## 📦 실행 파일 다운로드

[Releases](../../releases) 에서 최신 버전 다운로드

압축 해제 후 `main.exe` 실행 (`hand_landmarker.task` 같은 폴더에 있어야 함)

---

## 🧰 기술 스택

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.11 |
| 손 인식 | MediaPipe Hand Landmarker |
| 영상 처리 | OpenCV |
| 렌더링 | NumPy, Pillow |
| 선 스무딩 | Bezier Curve (Catmull-Rom) |

---

## 📁 압축 해제시 폴더 구조

```
water_hand/
├── main.py                # 메인 코드
├── hand_landmarker.task   # MediaPipe 모델
├── saves/                 # 저장된 캡처 이미지
```

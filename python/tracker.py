#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

HAND_CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)
POSE_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)

GREEN = (0, 230, 118)
RED = (68, 68, 255)
BOX_HOT = (73, 81, 248)
BOX_OK = (80, 185, 63)
WHITE = (255, 255, 255)
CYAN = (216, 180, 157)


def dist(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def finger_extended(lm, tip: int, pip: int) -> bool:
    return dist(lm[tip], lm[0]) > dist(lm[pip], lm[0]) * 1.12


def classify_hand(lm) -> str:
    thumb = dist(lm[4], lm[0]) > dist(lm[3], lm[0]) * 1.05
    index = finger_extended(lm, 8, 6)
    middle = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pinky = finger_extended(lm, 20, 18)
    raised = sum([index, middle, ring, pinky])
    if raised >= 4 and thumb:
        return "OPEN_PALM"
    if index and middle and not ring and not pinky:
        return "PEACE"
    if raised == 0 and not thumb:
        return "FIST"
    return "NONE"


class MotionDetector:
    def __init__(self, size=(96, 54), pixel_threshold=28):
        self.size = size
        self.pixel_threshold = pixel_threshold
        self.prev = None

    def reset(self):
        self.prev = None

    def process(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, self.size)
        if self.prev is None:
            self.prev = small
            return 0, None
        diff = cv2.absdiff(small, self.prev)
        moved = diff > self.pixel_threshold
        self.prev = small
        pct = int(moved.mean() * 100)
        if pct == 0:
            return 0, None
        ys, xs = np.where(moved)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        w, h = self.size
        box = (x0 / w, y0 / h, (x1 - x0 + 1) / w, (y1 - y0 + 1) / h)
        return pct, box


def draw_landmarks(img, points, connections, mirror: bool):
    h, w = img.shape[:2]

    def xy(p):
        x = int((1 - p.x) * w) if mirror else int(p.x * w)
        y = int(p.y * h)
        return x, y

    for a, b in connections:
        if a < len(points) and b < len(points):
            cv2.line(img, xy(points[a]), xy(points[b]), GREEN, 2, cv2.LINE_AA)
    for p in points:
        cv2.circle(img, xy(p), 4, RED, -1, cv2.LINE_AA)


def save_snapshot(frame) -> Path:
    path = SNAPSHOT_DIR / f"tracker_{int(time.time() * 1000)}.png"
    cv2.imwrite(str(path), frame)
    return path


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    motion = MotionDetector()

    running = True
    show_hands = True
    show_body = True
    show_motion = True
    show_gestures = True
    mirror = True
    sensitivity = 12
    last_gesture = "NONE"
    last_gesture_at = 0.0
    hold_name = "NONE"
    hold_since = 0.0
    status = "Tracking on"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            view = cv2.flip(frame, 1) if mirror else frame.copy()
            rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            overlay = view.copy()
            h, w = overlay.shape[:2]
            hand_count = 0
            body_found = False
            gesture = "NONE"
            pct = 0

            if running:
                if show_hands or show_gestures:
                    hand_res = hands.process(rgb)
                    if hand_res.multi_hand_landmarks:
                        hand_count = len(hand_res.multi_hand_landmarks)
                        for hlm in hand_res.multi_hand_landmarks:
                            if show_hands:
                                draw_landmarks(overlay, hlm.landmark, HAND_CONNECTIONS, False)
                            if show_gestures:
                                g = classify_hand(hlm.landmark)
                                if g != "NONE":
                                    gesture = g
                if show_body:
                    pose_res = pose.process(rgb)
                    if pose_res.pose_landmarks:
                        body_found = True
                        draw_landmarks(overlay, pose_res.pose_landmarks.landmark, POSE_CONNECTIONS, False)
                if show_motion:
                    pct, box = motion.process(view)
                    if box is not None:
                        x, y, bw, bh = box
                        x0, y0 = int(x * w), int(y * h)
                        x1, y1 = int((x + bw) * w), int((y + bh) * h)
                        color = BOX_HOT if pct >= sensitivity else BOX_OK
                        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
                now = time.time()
                if gesture != hold_name:
                    hold_name = gesture
                    hold_since = now
                    stable = "NONE"
                else:
                    stable = gesture if (gesture != "NONE" and now - hold_since >= 0.45) else "NONE"
                if show_gestures and stable != "NONE" and now - last_gesture_at > 1.2:
                    last_gesture_at = now
                    last_gesture = stable
                    if stable == "PEACE":
                        status = f"Snapshot {save_snapshot(overlay).name}"
                    elif stable == "FIST":
                        running = False
                        status = "Stopped (fist)"
                    elif stable == "OPEN_PALM":
                        running = True
                        status = "Started (palm)"
            else:
                cv2.putText(overlay, "PAUSED — show OPEN PALM to start",
                            (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

            y = 28
            for line in [
                f"{'RUN' if running else 'PAUSE'}  hands:{hand_count}  body:{'yes' if body_found else 'no'}",
                f"motion:{pct}%  sens:{sensitivity}  gesture:{last_gesture}",
                "palm=start  peace=shot  fist=stop",
                status,
            ]:
                cv2.putText(overlay, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2, cv2.LINE_AA)
                y += 26

            cv2.imshow("Motion + Body Tracker", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                status = f"Snapshot {save_snapshot(overlay).name}"
            elif key == ord("f"):
                mirror = not mirror
            elif key == ord("h"):
                show_hands = not show_hands
            elif key == ord("b"):
                show_body = not show_body
            elif key == ord("m"):
                show_motion = not show_motion
            elif key == ord("g"):
                show_gestures = not show_gestures
            elif key in (ord("+"), ord("=")):
                sensitivity = min(40, sensitivity + 1)
            elif key in (ord("-"), ord("_")):
                sensitivity = max(3, sensitivity - 1)
            elif key == ord(" "):
                running = not running
                motion.reset()
                status = "Tracking on" if running else "Paused"
    finally:
        cap.release()
        hands.close()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

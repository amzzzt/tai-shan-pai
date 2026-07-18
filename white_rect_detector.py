"""白色矩形检测模块"""

import cv2
import numpy as np


class WhiteRectDetector:
    def __init__(self):
        self.prev_cx, self.prev_cy = 0, 0
        self.locked = False
        self.lock_miss = 0
        self.found = False
        self.cx = self.cy = 0
        self.rect_pts = None
        self.dx = self.dy = 0
        self.mask = None

    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, np.array([79, 10, 134]), np.array([240, 128, 237]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        self.mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw < 8 or rh < 8:
                continue
            if area / (rw * rh) < 0.3:
                continue
            hull = cv2.convexHull(cnt)
            ha = cv2.contourArea(hull)
            if ha < 1 or area / ha < 0.5:
                continue
            ar = max(rw, rh) / min(rw, rh)
            if ar < 1.1 or ar > 2.0:
                continue
            if area > best_area:
                best_area = area
                best = rect

        if best is not None:
            (cx, cy), (rw, rh), angle = best
            if self.locked:
                dist = np.sqrt((cx - self.prev_cx)**2 + (cy - self.prev_cy)**2)
                if dist > 200:
                    best = None
            if best is not None:
                self.cx, self.cy = int(cx), int(cy)
                self.rect_pts = cv2.boxPoints(best).astype(np.int32)
                self.found = True
                self.lock_miss = 0
                self.locked = True
                self.prev_cx, self.prev_cy = self.cx, self.cy
                self.dx = self.cx - cx0
                self.dy = self.cy - cy0

        if best is None:
            self.found = False
            self.lock_miss += 1
            if self.lock_miss > 20:
                self.locked = False
            self.dx = self.dy = 0

        return self.found

    def draw(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2
        cv2.line(frame, (cx0 - 15, cy0), (cx0 + 15, cy0), (255, 255, 255), 2)
        cv2.line(frame, (cx0, cy0 - 15), (cx0, cy0 + 15), (255, 255, 255), 2)
        if self.found:
            cv2.polylines(frame, [self.rect_pts], True, (0, 255, 0), 2)
            cv2.line(frame, (cx0, cy0), (self.cx, self.cy), (0, 255, 255), 2)
            cv2.putText(frame, "dx=%+d dy=%+d" % (self.dx, self.dy),
                        (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No rect", (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

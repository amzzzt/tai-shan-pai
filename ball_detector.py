"""
钢球检测器
==========
球半径 10~30px, 只在Y轴中心±40范围内沿X轴运动
处理区域: 只处理画面中央水平带, 其余全部忽略
"""

import cv2
import numpy as np


class BallDetector:
    def __init__(self):
        self.found = False
        self.cx = self.cy = 0
        self.dx = self.dy = 0
        self.radius = 0.0
        self.mask = None

        self.min_area = 300     # r≈10: π×10²=314
        self.max_area = 2800    # r≈30: π×30²=2827
        self.detect_band = 20   # 球只在Y轴±20内, 管壁反光在外面
        self.dx_sign = -1

    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        # 1. 灰度 + 预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        tval = float(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[0])

        _, mask_full = cv2.threshold(gray, tval, 255, cv2.THRESH_BINARY_INV)
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # 2. 裁切mask: 只在Y±30内找球
        y1 = max(0, cy0 - self.detect_band)
        y2 = min(h, cy0 + self.detect_band)
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.mask[y1:y2, :] = mask_full[y1:y2, :]

        # 3. 找轮廓: 小 + 圆
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue
            if 4 * np.pi * area / (peri * peri) < 0.5:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            if best is None or abs(area - 1257) < abs(best[0] - 1257):
                best = (area, cx, cy, radius)

        if best is not None:
            _, self.cx, self.cy, self.radius = best
            self.cx, self.cy = int(self.cx), int(self.cy)
            self.dx = self.dx_sign * (self.cx - cx0)
            self.dy = 0
            self.found = True
        else:
            self.found = False
            self.dx = self.dy = 0
            self.radius = 0.0

        return self.found

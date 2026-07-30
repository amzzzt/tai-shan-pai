"""
钢球检测器 — 最简版
====================
水管=白, 球=黑 → 反相 → 球=白 → 找最大白块
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

        self.min_area = 150     # r≈7
        self.max_area = 500     # r≈12.6, 球r≈10
        self.detect_band = 20
        self.dx_sign = 1

        # 简单锁: 丢球时预测
        self.locked = False
        self.lost_frames = 0
        self.max_lost = 15
        self.last_cx = 0
        self.last_vx = 0.0

        self.kf_vx = 0.0
        self.kf_kx = 0.0
        self.noise_boost = 1.0

    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        # 1. OTSU反二值化 → 管=黑, 球=白
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask_full = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # 2. 裁到±20: 检测mask(管黑球白) + 显示mask(管白球黑)
        y1 = max(0, cy0 - self.detect_band)
        y2 = min(h, cy0 + self.detect_band)
        detect_mask = np.zeros((h, w), dtype=np.uint8)
        detect_mask[y1:y2, :] = mask_full[y1:y2, :]
        self.mask = cv2.bitwise_not(detect_mask)  # channel1: 白管黑球
        self.mask[:y1, :] = 0
        self.mask[y2:, :] = 0

        # 3. 在检测mask上找最大白块 = 球
        contours, _ = cv2.findContours(detect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < 6 or radius > 16:   # 球r≈10, 严格限制
                continue
            if area > best_area:
                best_area = area
                best = (cx, cy, radius)

        # 4. 状态
        if best is not None:
            cx, cy, self.radius = best
            if self.locked:
                self.last_vx = self.last_vx * 0.5 + (cx - self.last_cx) * 0.5
            self.last_cx = cx
            self.lost_frames = 0
            self.locked = True

        elif self.locked and self.lost_frames < self.max_lost:
            self.lost_frames += 1
            self.last_cx += self.last_vx
            cx, cy = self.last_cx, 0

        else:
            self.locked = False
            self.lost_frames = 0
            self.last_vx = 0.0
            self.found = False
            self.dx = self.dy = 0
            self.radius = 0.0
            return False

        self.cx, self.cy = int(cx), cy0
        self.dx = int(self.dx_sign * (cx - cx0))
        self.kf_kx = cx
        self.kf_vx = self.last_vx
        self.found = True
        return True

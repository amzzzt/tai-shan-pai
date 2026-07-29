"""
钢球检测器 + 简单追踪锁
========================
球半径 10~30px, 只在Y轴中心±20内沿X轴运动
丢球时用最后速度预测位置, ≤15帧内继续追
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

        self.min_area = 300
        self.max_area = 2800
        self.detect_band = 20
        self.dx_sign = 1

        # 追踪锁 - 位置
        self.locked = False
        self.lost_frames = 0
        self.max_lost = 15
        self.last_cx = 0
        self.last_cy = 0
        self.last_vx = 0.0

        # 追踪锁 - 大小
        self.last_radius = 0.0
        self.max_area_jump = 1.3  # 面积突增超过此倍率, 拒绝

    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        # 1. 灰度 + 预处理 (全帧, 球直径可能超检测带宽度)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        tval = float(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[0])

        _, mask_full = cv2.threshold(gray, tval, 255, cv2.THRESH_BINARY_INV)
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # 2. 裁切mask: 只在Y±20内找球
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

            # 锁定后: 位置+大小双重校验
            if self.locked:
                # 大小突变跳过 (阴影会让球变大)
                if self.last_radius > 0:
                    last_area = np.pi * self.last_radius ** 2
                    if area > last_area * self.max_area_jump:
                        continue
                    if area < last_area / self.max_area_jump:
                        continue
                # 丢帧中: 靠近预测位置的优先
                if self.lost_frames > 0:
                    pred_cx = self.last_cx + self.last_vx * self.lost_frames
                    if abs(cx - pred_cx) > 100:
                        continue

            if best is None or abs(area - 1257) < abs(best[0] - 1257):
                best = (area, cx, cy, radius)

        # 4. 追踪锁状态机
        if best is not None:
            _, cx, cy, self.radius = best
            # 更新速度 + 更新大小 (EMA平滑)
            if self.locked and self.lost_frames == 0:
                self.last_vx = self.last_vx * 0.4 + (cx - self.last_cx) * 0.6  # 60%新值, 快速响应
                self.last_radius = self.last_radius * 0.7 + self.radius * 0.3
            else:
                self.last_radius = self.radius
            self.last_cx = cx
            self.last_cy = cy
            self.lost_frames = 0
            self.locked = True
        elif self.locked and self.lost_frames < self.max_lost:
            # 丢球但锁住: 用速度预测
            self.lost_frames += 1
            self.last_cx += self.last_vx
            # 预测位置当作当前检测结果
            cx, cy = self.last_cx, self.last_cy
            self.radius = self.radius  # 保持上次半径
        else:
            # 彻底丢了, 解锁
            self.locked = False
            self.lost_frames = 0
            self.last_vx = 0.0
            self.found = False
            self.dx = self.dy = 0
            self.radius = 0.0
            return False

        # 5. 输出
        self.cx, self.cy = int(cx), int(cy)
        self.dx = self.dx_sign * (self.cx - cx0)
        self.dy = 0
        self.found = True
        return True

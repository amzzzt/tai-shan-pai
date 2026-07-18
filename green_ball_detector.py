"""绿色网球检测模块"""

import cv2
import numpy as np


class GreenBallDetector:
    def __init__(self):
        # LAB 阈值 (OpenMV 格式)
        self.set_thresholds([15, 79, -48, -5, 19, 104])

        # 追踪锁
        self.prev_cx, self.prev_cy, self.prev_r = 0, 0, 0
        self.locked = False
        self.lock_miss = 0

        # 结果
        self.found = False
        self.cx = 0
        self.cy = 0
        self.r = 0
        self.dx = 0
        self.dy = 0
        self.error_x = 0.0
        self.error_y = 0.0
        self.mask = None

    def set_thresholds(self, openmv_thresholds):
        l_min, l_max, a_min, a_max, b_min, b_max = openmv_thresholds
        self.lower = np.array([int(l_min * 2.55), int(a_min + 128), int(b_min + 128)])
        self.upper = np.array([int(l_max * 2.55), int(a_max + 128), int(b_max + 128)])

    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        # 1. LAB 掩码（先中值滤波去噪，减少形态学负担）
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, self.lower, self.upper)

        # 2. 形态学：开运算去噪点 + 闭运算填洞
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        self.mask = mask

        # 3. 轮廓筛选
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue

            circ = (4.0 * np.pi * area) / (peri * peri)
            if circ < 0.5:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1:
                continue
            solidity = area / hull_area
            if solidity < 0.65:
                continue

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            cx, cy, r = int(cx), int(cy), int(r)
            if r < 10:
                continue
            fill = area / (np.pi * r * r)
            if fill < 0.25:
                continue

            # 得分：半径的平方 × 圆形度 — 大圆严重优先
            score = (r * r) * circ * solidity * fill
            if self.locked:
                dist = np.sqrt((cx - self.prev_cx)**2 + (cy - self.prev_cy)**2)
                search_r = max(self.prev_r * 4, 80)
                if dist > search_r:
                    continue
                score += (search_r - dist) * 100
            if score > best_score:
                best_score = score
                best = (cx, cy, r)

        # 4. RGB 颜色验证（9 点稀疏采样）
        if best is not None:
            cx, cy, r = best
            sr = max(r - 3, 3)
            offsets = [(0,0), (-sr,0), (sr,0), (0,-sr), (0,sr),
                       (-sr//2,-sr//2), (sr//2,-sr//2), (-sr//2,sr//2), (sr//2,sr//2)]
            green_cnt = 0
            total = 0
            for ox, oy in offsets:
                px, py = cy + oy, cx + ox
                if 0 <= px < h and 0 <= py < w:
                    b, g, r_ch = frame[px, py].astype(float)
                    if g > r_ch * 1.15 and g > b * 1.10:
                        green_cnt += 1
                    total += 1
            if total > 0 and green_cnt / total < 0.3:
                best = None

        # 5. 更新状态
        if best is not None:
            cx, cy, r = best
            self.found = True
            self.cx, self.cy, self.r = cx, cy, r
            self.lock_miss = 0
            self.locked = True
            self.prev_cx, self.prev_cy, self.prev_r = cx, cy, r
            self.dx = cx - cx0
            self.dy = cy - cy0
            self.error_x = self.dx / cx0 if cx0 > 0 else 0.0
            self.error_y = self.dy / cy0 if cy0 > 0 else 0.0
        else:
            self.found = False
            self.lock_miss += 1
            if self.lock_miss > 20:
                self.locked = False
            self.dx = 0
            self.dy = 0
            self.error_x = 0.0
            self.error_y = 0.0

        return self.found

    def draw(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        # 中心十字
        cv2.line(frame, (cx0 - 15, cy0), (cx0 + 15, cy0), (255, 255, 255), 2)
        cv2.line(frame, (cx0, cy0 - 15), (cx0, cy0 + 15), (255, 255, 255), 2)

        if self.found:
            cv2.circle(frame, (self.cx, self.cy), self.r, (0, 255, 0), 3)
            cv2.circle(frame, (self.cx, self.cy), 4, (0, 255, 0), -1)
            cv2.line(frame, (cx0, cy0), (self.cx, self.cy), (0, 255, 255), 2)
            cv2.putText(frame, "dx=%+d dy=%+d" % (self.dx, self.dy),
                        (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No ball", (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

"""白色矩形边框检测模块"""

import cv2
import numpy as np


class BlackRectDetector:
    def __init__(self):
        self.lower = np.array([0, 90, 70])
        self.upper = np.array([18, 140, 160])
        self.prev_cx, self.prev_cy = 0, 0
        self.locked = False
        self.lock_miss = 0
        self.found = False
        self.cx = self.cy = 0
        self.outer_pts = None
        self.inner_pts = None
        self.dx = self.dy = 0
        self.mask = None

    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        # 1. 掩码 + 形态学
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, self.lower, self.upper)
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        # 膨胀连接断断续续的碎片
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        self.mask = mask

        # 2. 轮廓 → 找匹配的矩形框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_outer = None
        best_scale = 0.85
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > h * w * 0.5:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw < 12 or rh < 12:
                continue
            ar = max(rw, rh) / min(rw, rh)
            if ar < 1.15 or ar > 1.7:
                continue

            # 凸性放宽（碎片拼接后可能不平滑）
            hull = cv2.convexHull(cnt)
            ha = cv2.contourArea(hull)
            if ha < 1 or area / ha < 0.4:
                continue

            # 找最佳内缩比例：内框黑 + 外环白 同时满足
            best_s = 0
            best_dual = 0
            for s in [0.88, 0.856, 0.83, 0.80]:
                irw, irh = rw * s, rh * s
                ipts = cv2.boxPoints(((cx, cy), (irw, irh), angle)).astype(np.int32)
                opts = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(np.int32)
                # 内框黑占比
                inner_m = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(inner_m, [ipts], 255)
                in_tot = cv2.countNonZero(inner_m)
                if in_tot < 50:
                    continue
                black = in_tot - cv2.countNonZero(cv2.bitwise_and(mask, inner_m))
                black_r = black / in_tot
                # 外环白占比
                ring_m = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(ring_m, [opts], 255)
                cv2.fillPoly(ring_m, [ipts], 0)
                ring_tot = cv2.countNonZero(ring_m)
                if ring_tot < 50:
                    continue
                white_r = cv2.countNonZero(cv2.bitwise_and(mask, ring_m)) / ring_tot
                # 双重得分：黑心+白环
                dual = black_r * white_r
                if dual > best_dual:
                    best_dual = dual
                    best_s = s

            if best_dual < 0.06:  # 0.3黑 × 0.15白 ≈ 0.045 起
                continue

            if area > best_area:
                best_area = area
                best_outer = rect
                best_scale = best_s

        # 3. 追踪锁
        if best_outer is not None:
            cx, cy = int(best_outer[0][0]), int(best_outer[0][1])
            if self.locked:
                dist = np.sqrt((cx - self.prev_cx)**2 + (cy - self.prev_cy)**2)
                if dist > 200:
                    best_outer = None

        # 4. 更新
        if best_outer is not None:
            (ocx, ocy), (orw, orh), angle = best_outer
            irw, irh = orw * best_scale, orh * best_scale
            self.cx, self.cy = int(ocx), int(ocy)
            self.outer_pts = cv2.boxPoints(((ocx, ocy), (orw, orh), angle)).astype(np.int32)
            self.inner_pts = cv2.boxPoints(((ocx, ocy), (irw, irh), angle)).astype(np.int32)
            self.found = True
            self.lock_miss = 0
            self.locked = True
            self.prev_cx, self.prev_cy = self.cx, self.cy
            self.dx = self.cx - cx0
            self.dy = self.cy - cy0
        else:
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
            cv2.polylines(frame, [self.outer_pts], True, (0, 255, 0), 2)
            cv2.polylines(frame, [self.inner_pts], True, (0, 255, 0), 2)
            cv2.circle(frame, (self.cx, self.cy), 4, (0, 255, 0), -1)
            cv2.line(frame, (cx0, cy0), (self.cx, self.cy), (0, 255, 255), 2)
            cv2.putText(frame, "dx=%+d dy=%+d" % (self.dx, self.dy),
                        (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No rect", (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

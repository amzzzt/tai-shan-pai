"""标靶检测 — 自适应阈值 + 矩形匹配"""

import cv2
import numpy as np


class TargetDetector:
    def __init__(self):
        self.found = False
        self.cx = self.cy = 0
        self.dx = self.dy = 0
        self.outer_pts = None
        self.inner_pts = None
        self.mask = None
        self.scale = 1.17
        self.prev_cx, self.prev_cy = 0, 0
        self.locked = False
        self.lock_miss = 0

    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2
        img_area = h * w

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        best, best_area = None, 0

        # 同时试 Otsu 正相和反相，取矩形质量最好的
        try:
            tval = float(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0])
        except Exception:
            tval = 128
        # 每个方向生成掩码
        m1 = cv2.threshold(gray, tval, 255, cv2.THRESH_BINARY)[1]
        m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m2 = cv2.bitwise_not(m1)
        self.mask = m1  # 默认显示正相掩码
        masks = [m1, m2]

        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < img_area / 40 or area > img_area / 3:
                    continue
                # 严格四边形拟合
                peri = cv2.arcLength(cnt, True)
                if peri < 10:
                    continue
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                rect = cv2.minAreaRect(cnt)
                (_, _), (rw, rh), _ = rect
                if rw < 6 or rh < 6:
                    continue
                box = cv2.boxPoints(rect).astype(np.int32)
                box_area = cv2.contourArea(box)
                if box_area < 1:
                    continue
                # 高填充率：必须是实心矩形
                fill = area / box_area
                if fill < 0.65:
                    continue
                # 严格长宽比 25.5:17.5 ≈ 1.46
                ar = max(rw, rh) / max(min(rw, rh), 1)
                if ar < 1.25 or ar > 1.7:
                    continue
                score = fill * area
                if score > best_area:
                    best_area = score
                    best = box
                    self.mask = mask

        if best is not None:
            cx, cy = best[:, 0].mean(), best[:, 1].mean()
            if self.locked:
                dist = np.sqrt((cx - self.prev_cx)**2 + (cy - self.prev_cy)**2)
                if dist > 200:
                    best = None

        if best is not None:
            rect = cv2.minAreaRect(best)
            (icx, icy), (iw, ih), angle = rect
            ow, oh = iw * self.scale, ih * self.scale
            self.outer_pts = cv2.boxPoints(((icx, icy), (ow, oh), angle)).astype(np.int32)
            self.inner_pts = best
            self.cx, self.cy = int(icx), int(icy)
            self.dx, self.dy = self.cx - cx0, self.cy - cy0
            self.found = True
            self.lock_miss = 0
            self.locked = True
            self.prev_cx, self.prev_cy = self.cx, self.cy
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
            cv2.circle(frame, (self.cx, self.cy), 5, (0, 255, 255), -1)
            cv2.line(frame, (cx0, cy0), (self.cx, self.cy), (0, 255, 255), 2)
            cv2.putText(frame, "dx=%+d dy=%+d" % (self.dx, self.dy),
                        (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No target", (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

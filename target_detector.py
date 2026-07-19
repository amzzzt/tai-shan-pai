"""标靶检测 — 轮廓矩形法"""

import cv2
import numpy as np


class TargetDetector:
    def __init__(self):
        self.found = False
        self.cx = self.cy = 0
        self.dx = self.dy = 0
        self.outer_pts = None
        self.inner_pts = None
        self.mid_pts = None
        self.mask = None
        self.scale = 1.17
        self.prev_cx, self.prev_cy = 0, 0
        self.locked = False
        self.lock_miss = 0

    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        self.mask = edges

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_corners = None
        best_area = 0
        img_area = h * w
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area / 80 or area > img_area / 3:
                continue
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if len(approx) < 4 or len(approx) > 6:
                continue
            # 取 hull 上最远的 4 个点作为角点
            rect = cv2.minAreaRect(hull)
            rw, rh = rect[1]
            if rw < 8 or rh < 8:
                continue
            ar = max(rw, rh) / max(min(rw, rh), 1)
            if ar < 1.15 or ar > 1.85:
                continue
            corners = cv2.boxPoints(rect)  # 矩形4角
            if area > best_area:
                best_area = area
                best_corners = corners

        if best_corners is not None:
            # 对角线的4个端点就是矩形4个角
            pts = best_corners.astype(np.float32)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            if self.locked:
                dist = np.sqrt((cx - self.prev_cx)**2 + (cy - self.prev_cy)**2)
                if dist > 200:
                    best_corners = None

            if best_corners is not None:
                rect = cv2.minAreaRect(best_corners)
                (icx, icy), (iw, ih), angle = rect
                ow, oh = iw * self.scale, ih * self.scale
                self.outer_pts = cv2.boxPoints(((icx, icy), (ow, oh), angle)).astype(np.int32)
                self.inner_pts = best_corners.astype(np.int32)
                self.mid_pts = ((self.outer_pts.astype(float) + self.inner_pts.astype(float)) * 0.5)
                for i in range(4):
                    dx_i = self.mid_pts[i][0] - icx
                    dy_i = self.mid_pts[i][1] - icy
                    n = np.sqrt(dx_i*dx_i + dy_i*dy_i) + 1e-6
                    self.mid_pts[i][0] += dx_i / n * 3
                    self.mid_pts[i][1] += dy_i / n * 3
                self.mid_pts = self.mid_pts.astype(np.int32)
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
        else:
            self.found = False
            self.lock_miss += 1
            if self.lock_miss > 20:
                self.locked = False
            self.dx = self.dy = 0

        return self.found

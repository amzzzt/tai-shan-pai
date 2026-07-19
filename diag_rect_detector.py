"""对角线法矩形检测 — 四角+对角线+白填充 双重判定"""

import cv2
import numpy as np


class DiagRectDetector:
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
        self.prev_rw, self.prev_rh = 0, 0
        self.locked = False
        self.lock_miss = 0

    # ── 判定1：对角线等长且互相平分 ──
    def _is_rect(self, pts):
        d1 = np.linalg.norm(pts[0] - pts[2])
        d2 = np.linalg.norm(pts[1] - pts[3])
        if d1 < 10 or d2 < 10:
            return False
        if abs(d1 - d2) / max(d1, d2) > 0.15:
            return False
        m1 = (pts[0] + pts[2]) / 2
        m2 = (pts[1] + pts[3]) / 2
        if np.linalg.norm(m1 - m2) > max(d1, d2) * 0.1:
            return False
        return True

    # ── 判定2：矩形内部白色 + 外边界暗色 ──
    def _is_filled(self, gray, mask, corners, tval, h, w):
        inner = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(inner, [corners.astype(np.int32)], 255)
        total = cv2.countNonZero(inner)
        if total < 100:
            return False
        white = cv2.countNonZero(cv2.bitwise_and(mask, inner))
        if white / total < 0.6:
            return False
        bright = (gray[inner == 255] > tval).sum() / total
        if bright < 0.5:
            return False
        cr, (wr, hr), ar = cv2.minAreaRect(corners)
        outer_r = cv2.boxPoints((cr, (wr * 1.3, hr * 1.3), ar)).astype(np.int32)
        inner_r = cv2.boxPoints((cr, (wr, hr), ar)).astype(np.int32)
        for side in range(4):
            seg = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([outer_r[side], outer_r[(side+1)%4],
                           inner_r[(side+1)%4], inner_r[side]], dtype=np.int32)
            cv2.fillPoly(seg, [pts], 255)
            sp = cv2.countNonZero(seg)
            if sp > 30:
                if (gray[seg == 255] < tval).sum() / sp < 0.15:
                    return False
        return True

    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2
        img_area = h * w

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        tval = float(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0])

        # Otsu 正反两路
        best_corners, best_score = None, 0
        for tv in [tval, 255 - tval]:
            mask = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < img_area / 20 or area > img_area / 3:
                    continue
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if rw < 10 or rh < 10:
                    continue
                ar = max(rw, rh) / max(min(rw, rh), 1)
                if ar < 1.15 or ar > 1.85:
                    continue
                corners = cv2.boxPoints(rect)
                # 双重判定
                if not self._is_rect(corners):
                    continue
                if not self._is_filled(gray, mask, corners, tval, h, w):
                    continue
                score = area
                if score > best_score:
                    best_score = score
                    best_corners = corners
                    self.mask = mask

        # 追踪锁
        if best_corners is not None:
            pts = best_corners.astype(np.float32)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            cr, (rw, rh), _ = cv2.minAreaRect(best_corners)
            if self.locked:
                dist = np.sqrt((cx - self.prev_cx)**2 + (cy - self.prev_cy)**2)
                size_chg = abs(rw * rh - self.prev_rw * self.prev_rh) / max(self.prev_rw * self.prev_rh, 1)
                if dist > 120 or size_chg > 0.5:
                    best_corners = None

        if best_corners is not None:
            rect = cv2.minAreaRect(best_corners)
            (icx, icy), (iw, ih), angle = rect
            self.inner_pts = best_corners.astype(np.int32)
            ow, oh = iw * self.scale, ih * self.scale
            self.outer_pts = cv2.boxPoints(((icx, icy), (ow, oh), angle)).astype(np.int32)
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
            self.prev_rw, self.prev_rh = iw, ih
        else:
            self.found = False
            self.lock_miss += 1
            if self.lock_miss > 20:
                self.locked = False
            self.dx = self.dy = 0

        return self.found

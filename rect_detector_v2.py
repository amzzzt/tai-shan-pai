"""
矩形检测器 V2 — 整合三种开源方案的精华
===========================================
来源:
  1. K230-Vision-System (OpenMV find_rects)
     → Douglas-Peucker 多边形逼近 + 四边形快速筛选
  2. license-plate-detect-recoginition-pytorch (RetinaFace)
     → NMS去重 + 亚像素角点回归思想
  3. qrcode_recognition-master (pyzbar/zbar)
     → 对角线/垂直度几何验证 + 角点精度

针对回字靶场景的优化:
  - 正反两路 Otsu 自适应阈值
  - 对角线等长且互相平分（pyzbar级验证）
  - 邻边垂直度检查（排除梯形误检）
  - 内外框同心 + 固定比例（回字靶强约束）
  - 锁定追踪（位置+尺寸双重门槛）
"""

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════

def rect_angle(corners):
    """计算矩形4个内角 (度), 返回 list[float]"""
    angles = []
    pts = corners.astype(np.float32)
    for i in range(4):
        v1 = pts[(i + 1) % 4] - pts[i]
        v2 = pts[(i - 1) % 4] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            angles.append(90.0)
            continue
        cos_a = np.dot(v1, v2) / (n1 * n2)
        cos_a = np.clip(abs(cos_a), 0.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_a)))
    return angles


def is_valid_rect(corners, diag_tol=0.12, angle_tol=10.0, min_side=10):
    """
    pyzbar 级几何验证：对角线 + 邻边垂直度
    """
    pts = corners.astype(np.float32)
    # 对角线等长且互相平分
    d1 = np.linalg.norm(pts[0] - pts[2])
    d2 = np.linalg.norm(pts[1] - pts[3])
    if d1 < min_side or d2 < min_side:
        return False
    if abs(d1 - d2) / max(d1, d2) > diag_tol:
        return False
    # 中点重合
    m1 = (pts[0] + pts[2]) / 2
    m2 = (pts[1] + pts[3]) / 2
    if np.linalg.norm(m1 - m2) > max(d1, d2) * 0.08:
        return False
    # 邻边垂直度（排除梯形/菱形）
    angles = rect_angle(corners)
    for a in angles:
        if abs(a - 90) > angle_tol:
            return False
    return True


# ═══════════════════════════════════════════════════════════
#  主检测器
# ═══════════════════════════════════════════════════════════

class RectDetectorV2:
    def __init__(self):
        # 公有接口（兼容现有 main.py）
        self.found = False
        self.cx = self.cy = 0
        self.dx = self.dy = 0
        self.outer_pts = None
        self.inner_pts = None
        self.mid_pts = None
        self.mask = None
        self.scale = 1.17          # 外框/内框比例
        self.diag_tol = 0.15       # 对角线等长容差
        self.angle_tol = 12.0      # 邻边垂直容差
        self.min_area_ratio = 1/300 # 最小面积占图像比（2m+ 可过）
        self.max_area_ratio = 0.5   # 标靶最大不超过半屏

        # 追踪锁
        self.prev_cx, self.prev_cy = 0, 0
        self.prev_rw, self.prev_rh = 0, 0
        self.prev_ar = 0
        self.prev_angle = 0
        self.locked = False
        self.lock_miss = 0
        self._smoothed = None  # EMA 平滑角点

        # 内部状态（调试/可视化用）
        self.debug_candidates = []
        self.debug_pairs = []

    # ── K230 风格: DP 多边形逼近 + 四边形筛选 ──
    def _find_quads_dp(self, mask, img_area):
        min_a = img_area * self.min_area_ratio
        max_a = img_area * self.max_area_ratio

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_a or area > max_a:
                continue

            # 凸性检查：凹进去的轮廓直接拒（面积 / 凸包面积 < 0.80）
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and area / hull_area < 0.80:
                continue

            # Douglas-Peucker 逼近（放宽精度适应远距+畸变）
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            # 严格逼近：直边 4 顶，弯边炸出很多顶点
            approx_strict = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            n_strict = len(approx_strict)
            n = len(approx)

            # 3 顶点 → 补角
            if n == 3:
                pts = approx.reshape(3, 2)
                edges = [(np.linalg.norm(pts[i] - pts[(i+1)%3]), i) for i in range(3)]
                _, long_i = max(edges, key=lambda x: x[0])
                j = (long_i + 1) % 3
                mid = (pts[long_i] + pts[j]) / 2.0
                pts = np.vstack([pts, mid.reshape(1, 2)])
                corners = pts.astype(np.float32)
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                area_ratio = area / max(rw * rh, 1)
                peri_ratio = peri / (2*(rw+rh)) if (rw+rh) > 0 else 9
                quads.append((corners, area, area_ratio, peri_ratio, n_strict))
                continue

            # 严格四边形
            if n == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                area_ratio = area / max(rw * rh, 1)
                peri_ratio = peri / (2*(rw+rh)) if (rw+rh) > 0 else 9
                quads.append((corners, area, area_ratio, peri_ratio, n_strict))
                continue

            # 5~8 边回退 minAreaRect
            if 5 <= n <= 8:
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if rw < 4 or rh < 4:
                    continue
                ar = max(rw, rh) / max(min(rw, rh), 1)
                if ar < 1.05 or ar > 2.0:
                    continue
                corners = cv2.boxPoints(rect)
                area_ratio = area / max(rw * rh, 1)
                peri_ratio = peri / (2*(rw+rh)) if (rw+rh) > 0 else 9
                quads.append((corners, area, area_ratio, peri_ratio, n_strict))

        return quads

    # ── 亚像素角点精化（轻量版，提帧率）──
    def _refine_corners(self, gray, corners):
        try:
            refined = cv2.cornerSubPix(
                gray,
                corners.astype(np.float32),
                winSize=(3, 3),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.05)
            )
            return refined
        except cv2.error:
            return corners

    # ── 填充验证：内部白 + 外环暗（标靶 vs 白墙的关键区分）──
    def _check_fill(self, gray, corners, h, w, strict=False):
        inner = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(inner, [corners.astype(np.int32)], 255)
        total = cv2.countNonZero(inner)
        if total < 30:
            return False, 0.0

        mean_in = float(gray[inner == 255].mean())
        if mean_in < 35:
            return False, 0.0

        # ── 外环暗度：自适应环宽，检查黑边框 ──
        cr, (wr, hr), ar = cv2.minAreaRect(corners)
        ring = max(3.0, min(14.0, max(wr, hr) * 0.25))
        outer = cv2.boxPoints((cr, (wr + ring, hr + ring), ar)).astype(np.int32)
        i_r   = cv2.boxPoints((cr, (wr, hr), ar)).astype(np.int32)

        ring_sum, ring_cnt = 0.0, 0
        dark_sides = 0
        for side in range(4):
            seg = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([outer[side], outer[(side+1)%4],
                            i_r[(side+1)%4], i_r[side]], dtype=np.int32)
            cv2.fillPoly(seg, [pts], 255)
            sp = cv2.countNonZero(seg)
            if sp > 4:
                m = float(gray[seg == 255].mean())
                ring_sum += m * sp
                ring_cnt += sp
                if m < mean_in * 0.80:
                    dark_sides += 1

        if ring_cnt == 0:
            return False, 0.0

        mean_out = ring_sum / ring_cnt

        # 外环比内部明显暗 → 有黑边框 → 是标靶
        if mean_out > mean_in * 0.78:
            return False, 0.0
        need_dark = 3 if strict else 2
        if dark_sides < need_dark:
            return False, 0.0

        contrast = (mean_in - mean_out) / 255.0
        score = max(0.2, min(contrast / 0.35, 1.0))
        return True, score

    # ── 主检测 ──
    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2
        img_area = h * w

        # 1. 预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        tval = float(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0])

        best_corners = None
        best_score = 0.0
        self.mask = None

        # 2. Otsu 正反两路（第一路找到就跳过）
        for tv in [tval, 255 - tval]:
            mask = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            if self.mask is None:
                self.mask = mask  # 始终有 mask 可显示

            quads = self._find_quads_dp(mask, img_area)
            if not quads:
                continue

            quads.sort(key=lambda x: x[1], reverse=True)
            quads = quads[:8]

            for corners, area, area_ratio, peri_ratio, n_strict in quads:
                # 边界排除：角点离画面边缘 < 5px → 残影，拒
                if (corners[:,0].min() < 5 or corners[:,0].max() > w - 5 or
                    corners[:,1].min() < 5 or corners[:,1].max() > h - 5):
                    continue
                # 严格逼近顶点数：直边 ≈4，弯边炸出 >6
                if n_strict > 6:
                    continue
                if peri_ratio > 1.30:
                    continue
                if area_ratio < 0.65:
                    continue
                if not is_valid_rect(corners, self.diag_tol, self.angle_tol):
                    continue
                fill_ok, fill_score = self._check_fill(gray, corners, h, w, strict=self.locked)
                if not fill_ok:
                    continue

                # ── 角度质量：灯 bloom 四个角远偏离 90° ──
                angles = rect_angle(corners)
                angle_dev = sum(abs(a - 90) for a in angles) / 4.0
                angle_quality = max(0.0, 1.0 - angle_dev / 12.0)
                if angle_quality < 0.25:  # 太不像矩形，直接拒
                    continue

                # ── 五项打分 ──
                rect = cv2.minAreaRect(corners)
                (cx_r, cy_r), (rw, rh), _ = rect
                ar = max(rw, rh) / max(min(rw, rh), 1)
                ar_score = 1.0 - min(abs(ar - 1.46) / 0.5, 1.0)
                rect_score = min(area_ratio, 1.0)

                if self.locked:
                    dist = np.sqrt((cx_r - self.prev_cx)**2 + (cy_r - self.prev_cy)**2)
                    max_move = max(self.prev_rw, self.prev_rh) * 2.0
                    pos_score = max(0, 1.0 - dist / max_move) if max_move > 0 else 0
                    score = fill_score*0.20 + ar_score*0.15 + rect_score*0.15 + angle_quality*0.20 + pos_score*0.30
                else:
                    score = fill_score*0.28 + ar_score*0.20 + rect_score*0.22 + angle_quality*0.30

                if score > best_score:
                    best_score = score
                    best_corners = corners
                    self.mask = mask

            if best_corners is not None:
                break

        # 3. 最低分门槛（锁定后更严，防背景误入）
        threshold = 0.35 if self.locked else 0.45
        if best_score < threshold:
            best_corners = None

        if best_corners is not None:
            best_corners = self._refine_corners(gray, best_corners)

        # 3.5 角点排序
        if best_corners is not None:
            pts = best_corners
            sy = pts[np.argsort(pts[:, 1])]
            top, bot = sy[:2], sy[2:]
            tl, tr = top[np.argsort(top[:, 0])]
            bl, br = bot[np.argsort(bot[:, 0])]
            best_corners = np.array([tl, tr, br, bl], dtype=np.float32)

            # EMA 轻量平滑
            if self._smoothed is not None:
                best_corners = best_corners * 0.8 + self._smoothed * 0.2
            self._smoothed = best_corners.copy()

        # 4. 追踪锁（宽松，不卡框）
        if best_corners is not None:
            pts = best_corners.astype(np.float32)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            cr, (rw, rh), _ = cv2.minAreaRect(best_corners)
            if self.locked:
                dist = np.sqrt((cx - self.prev_cx)**2 + (cy - self.prev_cy)**2)
                max_move = max(self.prev_rw, self.prev_rh) * 0.8
                size_chg = abs(rw * rh - self.prev_rw * self.prev_rh) / max(self.prev_rw * self.prev_rh, 1)
                ar_cur = max(rw, rh) / max(min(rw, rh), 1)
                ar_chg = abs(ar_cur - self.prev_ar)
                if dist > max_move or size_chg > 0.4 or ar_chg > 0.30:
                    best_corners = None

        # 5. 更新状态
        if best_corners is not None:
            rect = cv2.minAreaRect(best_corners)
            (icx, icy), (iw, ih), angle = rect

            self.inner_pts = best_corners.astype(np.int32)
            ow, oh = iw * self.scale, ih * self.scale
            outer_raw = cv2.boxPoints(((icx, icy), (ow, oh), angle))
            # 排序 outer_pts 与 inner_pts [TL,TR,BR,BL] 对齐
            sy_o = outer_raw[np.argsort(outer_raw[:, 1])]
            top_o, bot_o = sy_o[:2], sy_o[2:]
            tl_o, tr_o = top_o[np.argsort(top_o[:, 0])]
            bl_o, br_o = bot_o[np.argsort(bot_o[:, 0])]
            self.outer_pts = np.array([tl_o, tr_o, br_o, bl_o], dtype=np.int32)

            # 中点计算（外扩一点点，向框外偏3像素）
            self.mid_pts = ((self.outer_pts.astype(float) + self.inner_pts.astype(float)) * 0.5)
            for i in range(4):
                dx_i = self.mid_pts[i][0] - icx
                dy_i = self.mid_pts[i][1] - icy
                n = np.sqrt(dx_i * dx_i + dy_i * dy_i) + 1e-6
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
            self.prev_ar = max(iw, ih) / max(min(iw, ih), 1)
            self.prev_angle = angle
        else:
            self.found = False
            self.lock_miss += 1
            if self.lock_miss > 20:
                self.locked = False
            self.dx = self.dy = 0

        return self.found


# ═══════════════════════════════════════════════════════════
#  测试（泰山派上用 WebStreamer，不要 cv2.imshow）
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time, os
    from xbhdcc_tools import WebStreamer

    os.system("fuser -k 8080/tcp /dev/video9 2>/dev/null")
    time.sleep(0.5)

    cap = cv2.VideoCapture(9, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    streamer = WebStreamer(port=8080)
    rd = RectDetectorV2()

    fps = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 0)
        h, w = frame.shape[:2]

        t0 = time.time()
        rd.detect(frame)
        dt = (time.time() - t0) * 1000

        # 绘制
        cv2.line(frame, (w//2-15, h//2), (w//2+15, h//2), (255,255,255), 2)
        cv2.line(frame, (w//2, h//2-15), (w//2, h//2+15), (255,255,255), 2)
        if rd.found:
            cv2.polylines(frame, [rd.inner_pts], True, (0,255,0), 2)
            cv2.circle(frame, (rd.cx, rd.cy), 4, (0,255,0), -1)
            cv2.putText(frame, "dx=%+d dy=%+d %.1fms" % (rd.dx, rd.dy, dt),
                       (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            cv2.putText(frame, "No rect %.1fms" % dt,
                       (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # FPS
        cv2.putText(frame, "fps: %.1f" % fps, (w-150, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        streamer.update_frame(0, frame)
        if rd.mask is not None:
            streamer.update_frame(1, rd.mask)

        curr = time.time()
        fps = (1/(curr-last_time))*0.3 + fps*0.7
        last_time = curr

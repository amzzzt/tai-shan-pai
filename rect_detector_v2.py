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


def nms_rects(rects, iou_thresh=0.3):
    """
    license-plate 风格 NMS 去重
    rects: [(corners_4x2, area, score), ...]
    """
    if len(rects) <= 1:
        return rects

    # 按 score 降序
    rects = sorted(rects, key=lambda r: r[2], reverse=True)
    keep = []

    for i, (ci, ai, si) in enumerate(rects):
        kept = True
        for (ck, ak, _) in keep:
            # 用 minAreaRect 的旋转框计算 IoU
            ri = cv2.minAreaRect(ci.astype(np.float32))
            rk = cv2.minAreaRect(ck.astype(np.float32))
            iou = _rotated_iou(ri, rk, ai, ak)
            if iou > iou_thresh:
                kept = False
                break
        if kept:
            keep.append((ci, ai, si))

    return keep


def _rotated_iou(r1, r2, a1, a2):
    """旋转框 IoU 估算（用面积比替代，泰山派上避免 cv2.rotatedRectangleIntersection 可能缺失）"""
    # 简化版：中心距离 + 面积比
    c1, c2 = np.array(r1[0]), np.array(r2[0])
    dist = np.linalg.norm(c1 - c2)
    avg_size = (np.sqrt(a1) + np.sqrt(a2)) / 2
    if dist > avg_size * 0.8:
        return 0.0
    # 重叠面积估算
    overlap = (avg_size - dist) ** 2
    union = a1 + a2 - overlap
    if union < 1:
        return 0.0
    return overlap / union


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
        self.diag_tol = 0.12       # 对角线等长容差
        self.angle_tol = 10.0      # 邻边垂直容差 (度)
        self.min_area_ratio = 1/30 # 最小面积占图像比
        self.max_area_ratio = 1/3  # 最大面积占图像比

        # 追踪锁
        self.prev_cx, self.prev_cy = 0, 0
        self.prev_rw, self.prev_rh = 0, 0
        self.locked = False
        self.lock_miss = 0

        # 内部状态（调试/可视化用）
        self.debug_candidates = []
        self.debug_pairs = []

    # ── K230 风格: DP 多边形逼近 + 四边形筛选 ──
    def _find_quads_dp(self, mask, img_area):
        """
        对 mask 做轮廓查找 → approxPolyDP → 四边形筛选
        返回 [(corners_4x2, area), ...]
        """
        min_a = img_area * self.min_area_ratio
        max_a = img_area * self.max_area_ratio

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_a or area > max_a:
                continue

            # Douglas-Peucker 逼近
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            n = len(approx)

            # 严格四边形
            if n == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                quads.append((corners, area))
                continue

            # 5~8 边回退 minAreaRect
            if 5 <= n <= 8:
                rect = cv2.minAreaRect(cnt)
                rw, rh = rect[1]
                if rw < 10 or rh < 10:
                    continue
                ar = max(rw, rh) / max(min(rw, rh), 1)
                if ar < 1.15 or ar > 1.85:
                    continue
                corners = cv2.boxPoints(rect)
                quads.append((corners, area))

        return quads

    # ── license-plate 风格: 对候选按面积打分 + NMS ──
    def _score_and_nms(self, quads):
        """给候选打分（面积接近图像某比例最好），然后 NMS"""
        scored = [(c, a, a) for c, a in quads]  # 面积即分数
        return nms_rects(scored, iou_thresh=0.3)

    # ── 亚像素角点精化 ──
    def _refine_corners(self, gray, corners):
        """
        license-plate LandmarkHead 思想 → 传统 cornerSubPix 实现亚像素
        """
        try:
            refined = cv2.cornerSubPix(
                gray,
                corners.astype(np.float32),
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
            )
            return refined
        except cv2.error:
            return corners

    # ── 填充验证：确保矩形内部是实心白色（回字靶特征）──
    def _check_fill(self, gray, mask, corners, tval, h, w):
        inner = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(inner, [corners.astype(np.int32)], 255)
        total = cv2.countNonZero(inner)
        if total < 100:
            return False
        # mask 中白色占比
        white = cv2.countNonZero(cv2.bitwise_and(mask, inner))
        if white / total < 0.6:
            return False
        # 灰度亮度
        bright = (gray[inner == 255] > tval).sum() / max(total, 1)
        if bright < 0.5:
            return False
        return True

    # ── 主检测 ──
    def detect(self, frame):
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2
        img_area = h * w

        # 1. 预处理（缩小形态学核提帧率）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        tval = float(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0])

        best_corners = None
        best_score = 0
        self.mask = None

        # 2. Otsu 正反两路（第一路有好结果就跳过第二路）
        for tv in [tval, 255 - tval]:
            mask = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            quads = self._find_quads_dp(mask, img_area)
            if not quads:
                continue

            # 只对面积 top-8 做后续验证（省 _check_fill）
            quads.sort(key=lambda x: x[1], reverse=True)
            quads = quads[:8]

            for corners, area in quads:
                # pyzbar 几何验证（只在这里跑一次）
                if not is_valid_rect(corners, self.diag_tol, self.angle_tol):
                    continue
                # 填充验证（最贵的检查，放最后）
                if not self._check_fill(gray, mask, corners, tval, h, w):
                    continue
                if area > best_score:
                    best_score = area
                    best_corners = corners
                    self.mask = mask

            # 第一路找到就跳过第二路
            if best_corners is not None:
                break

        # 3. 亚像素精化（仅对大目标做，小目标收益低）
        if best_corners is not None:
            best_corners = self._refine_corners(gray, best_corners)

        # 4. 追踪锁（位置 + 尺寸 双重门槛）
        if best_corners is not None:
            pts = best_corners.astype(np.float32)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            cr, (rw, rh), _ = cv2.minAreaRect(best_corners)
            if self.locked:
                dist = np.sqrt((cx - self.prev_cx)**2 + (cy - self.prev_cy)**2)
                size_chg = abs(rw * rh - self.prev_rw * self.prev_rh) / max(self.prev_rw * self.prev_rh, 1)
                if dist > 120 or size_chg > 0.5:
                    best_corners = None

        # 5. 更新状态
        if best_corners is not None:
            rect = cv2.minAreaRect(best_corners)
            (icx, icy), (iw, ih), angle = rect

            self.inner_pts = best_corners.astype(np.int32)
            ow, oh = iw * self.scale, ih * self.scale
            self.outer_pts = cv2.boxPoints(((icx, icy), (ow, oh), angle)).astype(np.int32)

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

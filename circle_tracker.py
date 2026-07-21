"""
圆形路径追踪 — 绕标靶第 3 个圈匀速走圆
==========================================
用法:
  ct = CircleTracker(rd)           # rd = RectDetectorV2 实例
  ct.start(period=10.0, n=36)      # 10 秒走一圈，36 个点
  while True:
      rd.detect(frame)
      pt = ct.update(h, w)         # 返回 (px, py, dx, dy) 或 None
      if pt:
          sc.send_error(dx, dy, True)
"""

import cv2
import numpy as np
import time


class CircleTracker:
    def __init__(self, detector):
        self.detector = detector       # RectDetectorV2，用来拿中心和尺寸
        self.radius_ratio = 6.0 / 17.5  # 第 3 圈半径 6cm / 内框短边 17.5cm ≈ 0.343
        self.n_points = 36
        self.period = 30.0             # 固定 30 秒一圈
        self.points = []               # [(px, py), ...]
        self.active = False
        self._start_time = 0
        self._last_send = 0
        self._lost_frames = 0          # 连续丢框帧数
        self._max_lost = 15            # 连续丢框超过此值 → 自动停追（≈0.5s @30fps）

    # ── 启动 ──
    def start(self, period=None, n_points=None):
        if period is not None:
            self.period = period
        if n_points is not None:
            self.n_points = n_points
        self._generate_path()
        self.active = True
        self._start_time = time.time()
        self._last_send = 0

    def stop(self):
        self.active = False
        self.points = []

    # ── 生成路径：以标靶中心为圆心，第 3 圈半径的 N 个点 ──
    def _generate_path(self):
        td = self.detector
        if td.inner_pts is None:
            self.points = []
            return

        # 内框短边 × ratio = 第 3 圈半径（像素）
        rect = cv2.minAreaRect(td.inner_pts.astype(np.float32))
        iw, ih = rect[1]
        radius = min(iw, ih) * self.radius_ratio

        cx, cy = td.cx, td.cy
        self.points = []
        for i in range(self.n_points):
            angle = 2.0 * np.pi * i / self.n_points
            px = cx + radius * np.cos(angle)
            py = cy + radius * np.sin(angle)
            self.points.append((px, py))

    # ── 每帧调用：返回当前目标点和偏差 ──
    def update(self, h, w):
        if not self.active:
            return None

        # 检测到新框时更新路径并清零丢帧计数
        if self.detector.found:
            self._generate_path()
            self._lost_frames = 0
        else:
            self._lost_frames += 1
            # 连续丢框超阈值 → 暂停发点（但时间继续走，路径保留）
            # 等重新识别到会用新位置 _generate_path() + 当前时间无缝接上
            if self._lost_frames > self._max_lost:
                return None

        if not self.points:
            return None

        # 按时间进度确定当前点（丢帧时间不停）
        elapsed = time.time() - self._start_time
        progress = (elapsed % self.period) / self.period
        idx = int(progress * self.n_points) % self.n_points

        px, py = self.points[idx]
        dx = int(px - w // 2)
        dy = int(py - h // 2)
        return (px, py, dx, dy)

    # ── 主动发送偏差到串口 ──
    def send(self, serial_comm, h, w, interval=0.05):
        """每 interval 秒发一次当前偏差"""
        now = time.time()
        if now - self._last_send < interval:
            return
        self._last_send = now

        pt = self.update(h, w)
        if pt is not None:
            _, _, dx, dy = pt
            serial_comm.send_error(dx, dy, True)
        else:
            serial_comm.send_error(0, 0, False)

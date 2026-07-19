"""扫描控制 — 动态追逐：始终发1/3点，直到逼近目标角再切下一个"""

import numpy as np
import time


class ScanController:
    def __init__(self, target_detector, serial_comm):
        self.td = target_detector
        self.sc = serial_comm
        self._corner_idx = 1      # 从TL出发后追TR(1), 然后BR(2), BL(3), TL(0)
        self._arrived = False     # 是否刚到达TL（第一步）
        self._last_send = 0
        self.target_pt = None

    def _send(self, px, py, h, w):
        now = time.time()
        if now - self._last_send < 0.3:
            return
        self._last_send = now
        self.sc.send_error(int(px - w // 2), int(py - h // 2), True)

    def update(self, h, w):
        self.target_pt = None
        if not self.td.found or self.td.outer_pts is None:
            return None, self._corner_idx

        # 用内外框中点（在白框上）作为角点
        pts = self.td.mid_pts if self.td.mid_pts is not None else self.td.outer_pts
        sy = pts[np.argsort(pts[:, 1])]
        top, bot = sy[:2], sy[2:]
        tl, tr = top[np.argsort(top[:, 0])]
        bl, br = bot[np.argsort(bot[:, 0])]
        corners = [tl, tr, br, bl]

        if not self._arrived:
            # 第一步：直奔左上角
            self.target_pt = corners[0]
            self._send(corners[0][0], corners[0][1], h, w)
            # 偏差小了就判定到达
            dx = int(corners[0][0] - w // 2)
            dy = int(corners[0][1] - h // 2)
            if abs(dx) < 20 and abs(dy) < 20:
                self._arrived = True
            return self.target_pt, 0

        # 之后：追逐1/3点 — 起点永远是画面中心，终点是当前目标角
        target = corners[self._corner_idx]

        # 判断是否已逼近目标角
        dx_t = int(target[0] - w // 2)
        dy_t = int(target[1] - h // 2)
        if abs(dx_t) < 20 and abs(dy_t) < 20:
            # 切到下一个角
            self._corner_idx = (self._corner_idx + 1) % 4

        # 发1/3点（画面中心到目标角的1/3处）
        cx, cy = w // 2, h // 2
        px = cx + (target[0] - cx) / 3.0
        py = cy + (target[1] - cy) / 3.0
        self.target_pt = (px, py)
        self._send(px, py, h, w)

        return self.target_pt, self._corner_idx + 10  # 状态号10~13区分角点

    def reset(self):
        self._corner_idx = 1
        self._arrived = False
        self.target_pt = None

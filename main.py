import cv2
import numpy as np
from xbhdcc_tools import WebStreamer
import time
import os
from xbhdcc_spi_lcd import ST7735Streamer
from gpio_button import GpioButton
from rect_detector_v2 import RectDetectorV2
from circle_tracker import CircleTracker
from serial_comm import SerialComm

if __name__ == "__main__":
    os.system("fuser -k 8080/tcp /dev/video9 2>/dev/null")
    time.sleep(0.5)

    cap = cv2.VideoCapture(9, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    streamer = WebStreamer(port=8080)
    rd = RectDetectorV2()
    ct = CircleTracker(rd)
    sc = SerialComm(port='/dev/ttyS7', baudrate=115200)
    lcd = ST7735Streamer()

    # ── GPIO 按键：按一下显示/关闭 FPS ──
    # 引脚配置: 传数字编号或 "GPIO3_A1" 字符串均可
    # 接线: 按键一脚接 GPIO3_A1(TN 物理 40Pin 左侧第 36 脚)，另一脚接 3.3V
    btn = GpioButton(97)   # 97 = GPIO3_A1
    show_fps = False

    fps = 0
    last_time = time.time()
    confirm = 0
    ct_started = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 0)
        h, w = frame.shape[:2]

        rd.detect(frame)

        if rd.found:
            confirm += 1
        else:
            confirm = 0

        confirmed = confirm >= 2

        # 确认后启动圆形追踪（只启动一次，丢帧不重置）
        if confirmed and not ct_started:
            ct.start(period=30.0, n_points=72)
            ct_started = True

        # 获取圆形路径当前点（即使丢帧也按时间推进）
        ct_pt = ct.update(h, w) if ct_started else None

        # 发偏差：只跟红圈
        if ct_pt is not None:
            _, _, dx, dy = ct_pt
            sc.send_error(dx, dy, True)
        else:
            sc.send_error(0, 0, False)

        # ── 绘制 ──
        cv2.line(frame, (w//2-15, h//2), (w//2+15, h//2), (255,255,255), 2)
        cv2.line(frame, (w//2, h//2-15), (w//2, h//2+15), (255,255,255), 2)

        if confirmed and rd.inner_pts is not None:
            # 绿色矩形框
            cv2.polylines(frame, [rd.inner_pts], True, (0, 255, 0), 2)
            cv2.line(frame, tuple(rd.inner_pts[0]), tuple(rd.inner_pts[2]), (255,0,0), 1)
            cv2.line(frame, tuple(rd.inner_pts[1]), tuple(rd.inner_pts[3]), (255,0,0), 1)

            # 圆形路径（青色虚线圆）
            if ct.points:
                pts_int = [(int(p[0]), int(p[1])) for p in ct.points]
                for i in range(len(pts_int)):
                    cv2.circle(frame, pts_int[i], 1, (255, 255, 0), -1)

            # 中心红点
            cv2.circle(frame, (rd.cx, rd.cy), 5, (0, 0, 255), -1)

        # 圆形路径上的移动红点
        if ct_pt is not None:
            px, py, dx, dy = ct_pt
            cv2.circle(frame, (int(px), int(py)), 8, (0, 0, 255), -1)
            cv2.circle(frame, (int(px), int(py)), 10, (0, 0, 255), 2)
            cv2.putText(frame, "CT: dx=%+d dy=%+d" % (dx, dy),
                        (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No target", (5, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # GPIO 按键：按一下切换 FPS 显示
        if btn.update():
            show_fps = not show_fps

        streamer.update_frame(0, frame)
        lcd.update_frame(frame, overlay=("FPS:%.1f" % fps) if show_fps else None)
        if rd.mask is not None:
            streamer.update_frame(1, rd.mask)

        curr_time = time.time()
        fps = (1/(curr_time-last_time))*0.3 + fps*0.7
        last_time = curr_time

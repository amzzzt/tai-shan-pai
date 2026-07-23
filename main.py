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

    cap = cv2.VideoCapture(10, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    streamer = WebStreamer(port=8080)
    rd = RectDetectorV2()
    ct = CircleTracker(rd)

    # 串口（未接则降级为 dummy）
    try:
        sc = SerialComm(port='/dev/ttyS3', baudrate=115200)
    except Exception as e:
        print(f"[main] 串口打开失败 ({e})，使用 dummy 模式")
        sc = None

    # SPI LCD（未接则降级为 dummy）
    try:
        lcd = ST7735Streamer()
    except Exception as e:
        print(f"[main] SPI LCD 打开失败 ({e})，使用 dummy 模式")
        lcd = None

    # ── GPIO 按键：长按 2 秒 暂停/恢复 ──
    # 接线: 按键一脚接 GPIO3_A1，另一脚接 3.3V
    try:
        btn = GpioButton(97)   # 97 = GPIO3_A1
    except Exception as e:
        print(f"[main] GPIO 打开失败 ({e})，按键功能禁用")
        btn = None
    paused = False
    hold_frames = 0            # 长按计数
    HOLD_THRESHOLD = 60        # 长按阈值（约2秒@30fps）
    HOLD_COOLDOWN = 30         # 触发后冷却，防重复触发

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

        # ── 长按检测：按住 2 秒 暂停/恢复 ──
        if btn and btn.raw():
            hold_frames += 1
            if hold_frames == HOLD_THRESHOLD:
                paused = not paused
                print(f"[main] {'暂停' if paused else '恢复'}追踪")
                hold_frames = -HOLD_COOLDOWN  # 冷却，防重复触发
        else:
            hold_frames = max(hold_frames, 0)  # 松开归零，但不低于冷却值
            if hold_frames < 0:
                hold_frames += 1

        if paused:
            # 暂停时只推流，不追踪不发串口
            cv2.putText(frame, "PAUSED", (w//2-50, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            streamer.update_frame(0, frame)
            if lcd:
                lcd.update_frame(frame)
            if rd.mask is not None:
                streamer.update_frame(1, rd.mask)
            curr_time = time.time()
            fps = (1/(curr_time-last_time))*0.3 + fps*0.7
            last_time = curr_time
            continue

        # 发偏差：只跟红圈
        if ct_pt is not None:
            _, _, dx, dy = ct_pt
            if sc:
                sc.send_error(dx, dy, True)
        else:
            if sc:
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

        # FPS 直接画在画面左上角
        cv2.putText(frame, "FPS:%.1f" % fps, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        streamer.update_frame(0, frame)
        if lcd:
            lcd.update_frame(frame)
        if rd.mask is not None:
            streamer.update_frame(1, rd.mask)

        curr_time = time.time()
        fps = (1/(curr_time-last_time))*0.3 + fps*0.7
        last_time = curr_time

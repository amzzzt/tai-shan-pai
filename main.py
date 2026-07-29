import cv2
import numpy as np
from xbhdcc_tools import WebStreamer
import time
import os
# from xbhdcc_spi_lcd import ST7735Streamer  # 提速: 暂时关掉LCD
from ball_detector import BallDetector
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
    bd = BallDetector()
    sc = SerialComm(port='/dev/ttyS7', baudrate=115200)
    # lcd = ST7735Streamer()

    fps = 0
    last_time = time.time()

    calib_radii = [20, 30, 50, 80]     # 少画几个圆, 够用
    calib_y_lines = [20, 40, 60]

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, -1)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.flip(frame, 1)
        cy_full = frame.shape[0] // 2
        frame = frame[cy_full - 70:cy_full + 70, :]
        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        bd.detect(frame)

        if sc:
            sc.send_error(bd.dx, bd.dy, bd.found)

        # ── 校准参考线 ──
        for yb in calib_y_lines:
            cv2.line(frame, (0, cy0 - yb), (w, cy0 - yb), (0, 255, 255), 1)
            cv2.line(frame, (0, cy0 + yb), (w, cy0 + yb), (0, 255, 255), 1)
        for r in calib_radii:
            cv2.circle(frame, (cx0, cy0), r, (255, 255, 255), 1)
            cv2.putText(frame, "r%d" % r, (cx0 + r + 2, cy0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        if bd.found:
            cv2.circle(frame, (bd.cx, bd.cy), int(bd.radius), (0, 255, 0), 1)
            cv2.circle(frame, (bd.cx, bd.cy), 3, (0, 0, 255), -1)
            cv2.putText(frame, "dx=%+d" % bd.dx,
                        (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No ball", (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, "FPS:%.1f" % fps, (w - 130, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        streamer.update_frame(0, frame)
        if bd.mask is not None:
            streamer.update_frame(1, bd.mask)

        curr_time = time.time()
        fps = (1 / (curr_time - last_time)) * 0.3 + fps * 0.7
        last_time = curr_time

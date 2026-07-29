import cv2
import numpy as np
from xbhdcc_tools import WebStreamer
import time
import os
from xbhdcc_spi_lcd import ST7735Streamer
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
    try:
        sc = SerialComm(port='/dev/ttyS7', baudrate=115200)
    except Exception:
        sc = None
    try:
        lcd = ST7735Streamer()
    except Exception:
        lcd = None

    fps = 0
    last_time = time.time()
    calibrate = True

    calib_radii = [10, 20, 30, 40, 50, 60, 80, 100]
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

        if calibrate:
            for r in calib_radii:
                cv2.circle(frame, (cx0, cy0), r, (255, 255, 255), 1)
                cv2.putText(frame, "r=%d" % r, (cx0 + r + 3, cy0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            for yb in calib_y_lines:
                cv2.line(frame, (0, cy0 - yb), (w, cy0 - yb), (0, 255, 255), 1)
                cv2.line(frame, (0, cy0 + yb), (w, cy0 + yb), (0, 255, 255), 1)
                cv2.putText(frame, "y=+-%d" % yb, (5, cy0 + yb - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        if bd.found:
            cv2.circle(frame, (bd.cx, bd.cy), int(bd.radius), (0, 255, 0), 2)
            cv2.circle(frame, (bd.cx, bd.cy), 5, (0, 0, 255), -1)
            cv2.line(frame, (cx0, cy0), (bd.cx, bd.cy), (0, 255, 255), 1)
            cv2.putText(frame, "dx=%+d r=%.0f" % (bd.dx, bd.radius),
                        (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No ball", (5, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(frame, "fps: %.1f" % fps, (w - 140, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        streamer.update_frame(0, frame)
        if lcd:
            lcd.update_frame(frame)
        if bd.mask is not None:
            streamer.update_frame(1, bd.mask)

        curr_time = time.time()
        fps = (1 / (curr_time - last_time)) * 0.3 + fps * 0.7
        last_time = curr_time

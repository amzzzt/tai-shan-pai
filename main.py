import cv2
from xbhdcc_tools import WebStreamer
import time
import os
from rect_detector_v2 import RectDetectorV2
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
    sc = SerialComm(port='/dev/ttyS7', baudrate=115200)

    fps = 0
    last_time = time.time()
    confirm = 0  # 连续确认计数

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 0)
        h, w = frame.shape[:2]

        rd.detect(frame)

        # 连续 2 帧检测到才确认，滤偶发跳变
        if rd.found:
            confirm += 1
        else:
            confirm = 0

        confirmed = confirm >= 2

        # 发偏差
        if confirmed:
            sc.send_error(rd.dx, rd.dy, True)
        else:
            sc.send_error(0, 0, False)

        # 中心十字
        cv2.line(frame, (w//2-15, h//2), (w//2+15, h//2), (255,255,255), 2)
        cv2.line(frame, (w//2, h//2-15), (w//2, h//2+15), (255,255,255), 2)

        if confirmed and rd.inner_pts is not None:
            cv2.polylines(frame, [rd.inner_pts], True, (0, 255, 0), 2)
            cv2.line(frame, tuple(rd.inner_pts[0]), tuple(rd.inner_pts[2]), (255,0,0), 1)
            cv2.line(frame, tuple(rd.inner_pts[1]), tuple(rd.inner_pts[3]), (255,0,0), 1)
            cv2.circle(frame, (rd.cx, rd.cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, "dx=%+d dy=%+d" % (rd.dx, rd.dy),
                        (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No target", (5, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "fps: %.1f" % fps, (w-150, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        streamer.update_frame(0, frame)
        if rd.mask is not None:
            streamer.update_frame(1, rd.mask)

        curr_time = time.time()
        fps = (1/(curr_time-last_time))*0.3 + fps*0.7
        last_time = curr_time

import cv2
from xbhdcc_tools import WebStreamer
import time
import os
from diag_rect_detector import DiagRectDetector
from rect_detector_v2 import RectDetectorV2
from serial_comm import SerialComm
from scan_controller import ScanController

STATE_NAMES = {0: "->TL", 10: "->TR", 11: "->BR", 12: "->BL", 13: "->TL"}

if __name__ == "__main__":
    os.system("fuser -k 8080/tcp /dev/video9 2>/dev/null")
    time.sleep(0.5)

    cap = cv2.VideoCapture(9, cv2.CAP_V4L2)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    streamer = WebStreamer(port=8080)
    td = DiagRectDetector()
    rd_v2 = RectDetectorV2()
    sc = SerialComm(port='/dev/ttyS7', baudrate=115200)
    scan = ScanController(td, sc)

    fps = 0
    frame_cnt = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 0)
        h, w = frame.shape[:2]

        # 隔帧检测 + rd_v2 优先（几何验证更严格）
        if frame_cnt % 2 == 0:
            rd_v2.detect(frame)
            if rd_v2.found:
                td.found = True
                td.cx, td.cy = rd_v2.cx, rd_v2.cy
                td.dx, td.dy = rd_v2.dx, rd_v2.dy
                td.inner_pts = rd_v2.inner_pts
                td.outer_pts = rd_v2.outer_pts
                td.mid_pts = rd_v2.mid_pts
                td.mask = rd_v2.mask
            else:
                td.detect(frame)
        frame_cnt += 1
        target_pt, state = scan.update(h, w)

        # 中心十字 + 内部绿色矩形框
        cv2.line(frame, (w//2 - 15, h//2), (w//2 + 15, h//2), (255, 255, 255), 2)
        cv2.line(frame, (w//2, h//2 - 15), (w//2, h//2 + 15), (255, 255, 255), 2)
        if td.found and td.inner_pts is not None:
            pts = td.inner_pts
            # 先画对角线（蓝色，在下方）
            cv2.line(frame, tuple(pts[0]), tuple(pts[2]), (255, 0, 0), 1)
            cv2.line(frame, tuple(pts[1]), tuple(pts[3]), (255, 0, 0), 1)
            # 再画矩形（绿色，在上方）
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        # 红色目标点
        if target_pt is not None:
            cv2.circle(frame, (int(target_pt[0]), int(target_pt[1])), 8, (0, 0, 255), -1)

        cv2.putText(frame, "state: {}".format(STATE_NAMES.get(state, state)), [w - 150, 30],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if target_pt is not None:
            dx_s = int(target_pt[0] - w // 2)
            dy_s = int(target_pt[1] - h // 2)
            cv2.putText(frame, "send: dx=%+d dy=%+d" % (dx_s, dy_s),
                        (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # FPS 右下角
        cv2.putText(frame, "fps: {}".format(round(fps, 2)), [w - 200, h - 10],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        streamer.update_frame(0, frame)
        streamer.update_frame(1, td.mask)

        curr_time = time.time()
        fps = (1 / (curr_time - last_time) * 0.3 + fps * 0.7)
        last_time = curr_time


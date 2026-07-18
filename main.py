import cv2
from xbhdcc_tools import WebStreamer
import time
import os
from green_ball_detector import GreenBallDetector
from serial_comm import SerialComm

if __name__ == "__main__":
    os.system("fuser -k 8080/tcp /dev/video9 2>/dev/null")
    time.sleep(0.5)

    cap = cv2.VideoCapture(9, cv2.CAP_V4L2)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    streamer = WebStreamer(port=8080)
    ball = GreenBallDetector()
    serial_comm = SerialComm(port='/dev/ttyS7', baudrate=115200)
    fps = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 0)
        ball.detect(frame)
        ball.draw(frame)
        serial_comm.send_error(ball.dx, ball.dy, ball.found)

        cv2.putText(frame, "fps: {}".format(round(fps, 2)), [50, 50],
                    cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 0, 0], 2)

        streamer.update_frame(0, frame)
        streamer.update_frame(1, ball.mask)

        curr_time = time.time()
        fps = (1 / (curr_time - last_time) * 0.3 + fps * 0.7)
        last_time = curr_time

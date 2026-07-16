import cv2
from xbhdcc_tools import detect_cameras,WebStreamer
import time
import numpy as np

#[31, 100, -128, 124, -128, 21]

def convert_lab_thresholds(openmv_thresholds):
    l_min, l_max, a_min, a_max, b_min, b_max = openmv_thresholds
    lower_bound = np.array([int(l_min * 2.55), int(a_min + 128), int(b_min + 128)])
    upper_bound = np.array([int(l_max * 2.55), int(a_max + 128), int(b_max + 128)])
    return lower_bound, upper_bound

if __name__ == "__main__":
    detect_cameras()
    #BGR
    cap = cv2.VideoCapture(9, cv2.CAP_V4L2)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC,fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FPS,60)

    streamer = WebStreamer(port=8080)
    fps = 0

    last_time = time.time()
    lower_, upper_ = convert_lab_thresholds([31,100,-128,124,-128,21])

    while True:
        ret ,frame = cap.read()
        if not ret :
            continue
        frame = frame[180:540 , 320:960]    #先y后x

        frame_drawn = frame.copy()

        cv2.putText(frame_drawn,"fps: {}".format(round(fps,2)),[50,50],cv2.FONT_HERSHEY_SIMPLEX,2,[255,0,0],2)

        frame_drawn = cv2.cvtColor(frame_drawn,cv2.COLOR_BGR2LAB)
        frame_drawn = cv2.inRange(frame_drawn,lower_,upper_)

        #去除小噪点
        kernel = np.ones((5, 5), np.uint8)
        frame_drawn = cv2.morphologyEx(frame_drawn, cv2.MORPH_OPEN, kernel)
        frame_drawn = cv2.morphologyEx(frame_drawn, cv2.MORPH_CLOSE, kernel)

        #检测边缘
        frame_drawn = cv2.Canny(frame_drawn,50,150)

        #检测多边形轮廓点
        contours, _ = cv2.findContours(frame_drawn, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
             # 计算面积
             area = cv2.contourArea(cnt)
             if area < 2000:
                  continue

             # 计算周长
             perimeter = cv2.arcLength(cnt, True)

             # 拟合
             approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
             if not len(approx) == 4:
                continue

             p_0 = approx[0][0] # [796,344]
             p_1 = approx[1][0] # [796,344]
             p_2 = approx[2][0] # [796,344]
             p_3 = approx[3][0] # [796,344]

             cv2.line(frame, p_0, p_1, (0, 255, 0), 2)
             cv2.line(frame, p_1, p_2, (0, 255, 0), 2)
             cv2.line(frame, p_2, p_3, (0, 255, 0), 2)
             cv2.line(frame, p_3, p_0, (0, 255, 0), 2)

        streamer.update_frame(0,frame)
        streamer.update_frame(1,frame_drawn)
        curr_time = time.time()
        fps = (1 / (curr_time - last_time) *0.3 + fps * 0.7)
        last_time = curr_time
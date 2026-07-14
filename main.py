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
    #绿色花露水瓶阈值：a通道取负值区（绿），可对照网页流微调
    lower_, upper_ = convert_lab_thresholds([15,95,-128,-12,-30,70])

    while True:
        ret ,frame = cap.read()
        if not ret :
            continue
        frame = frame[180:540 , 320:960]    #先y后x

        mask = cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(mask,lower_,upper_)

        #去除小噪点
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        #直接在二值掩码上找外轮廓（不需要Canny，轮廓封闭更完整）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #在杂物中挑出最像花露水瓶的目标：面积最大的竖长绿色块
        best = None
        best_area = 0
        for cnt in contours:
             # 计算面积
             area = cv2.contourArea(cnt)
             if area < 2000:
                  continue

             x, y, w, h = cv2.boundingRect(cnt)

             # 花露水瓶是竖长的，过滤掉扁平的绿色杂物（横放请去掉此判断）
             if h < w * 1.2:
                  continue

             if area > best_area:
                  best_area = area
                  best = (x, y, w, h)

        if best is not None:
             x, y, w, h = best
             #红色框(BGR)框出花露水
             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
             cv2.putText(frame, "HuaLuShui", (x, max(y - 10, 20)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame,"fps: {}".format(round(fps,2)),[50,50],cv2.FONT_HERSHEY_SIMPLEX,2,[255,0,0],2)

        streamer.update_frame(0,frame)
        streamer.update_frame(1,mask)
        curr_time = time.time()
        fps = (1 / (curr_time - last_time) *0.3 + fps * 0.7)
        last_time = curr_time
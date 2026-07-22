from periphery import GPIO
import time

# 泰山派 40Pin 引脚编号速查字典
GPIO_MAP = {
    # GPIO0 组
    "GPIO0_B5": 13,
    "GPIO0_B6": 14,
    "GPIO0_B7": 15,
    
    # GPIO1 组
    "GPIO1_A4": 36,
    
    # GPIO3 组 (A组)
    "GPIO3_A1": 97,
    "GPIO3_A2": 98,
    "GPIO3_A3": 99,
    "GPIO3_A4": 100,
    "GPIO3_A5": 101,
    "GPIO3_A6": 102,
    "GPIO3_A7": 103,
    
    # GPIO3 组 (B组)
    "GPIO3_B0": 104,
    "GPIO3_B1": 105,
    "GPIO3_B2": 106,
    "GPIO3_B3": 107,
    "GPIO3_B4": 108,
    "GPIO3_B5": 109,
    "GPIO3_B6": 110,
    "GPIO3_B7": 111,
    
    # GPIO3 组 (C组)
    "GPIO3_C0": 112,
    "GPIO3_C2": 114,
    "GPIO3_C3": 115,
    "GPIO3_C4": 116,
    "GPIO3_C5": 117,
    
    # GPIO4 组
    "GPIO4_C2": 146,
    "GPIO4_C3": 147,
    "GPIO4_C5": 149,
    "GPIO4_C6": 150,
}
        


import cv2
from xbhdcc_tools import detect_cameras, WebStreamer
import time
import numpy as np
from my_tools import convert_lab_thresholds, my_draw_rect, find_rects, find_colors

# [20, 66, 11, 127, -50, 70]

if __name__ == "__main__":
    # detect_cameras()
    # BGR
    tmp_aaaa = 0
    cap = cv2.VideoCapture(9, cv2.CAP_V4L2)

    # 设置相机数据模式
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # 设置分辨率和帧率（不能随便填，要根据实际情况）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    streamer = WebStreamer(port=8081)
    fps = 0

    last_time = time.time()

    # 转换格式
    lower_color, upper_color = convert_lab_thresholds([20, 66, 11, 127, -50, 70])
    lower_rect, upper_rect = convert_lab_thresholds([44, 100, -26, 27, -24, 28])

    KEY_10_GPIO = GPIO_MAP["GPIO3_A1"]
    KEY_10 = GPIO(KEY_10_GPIO, "in")
    KEY_10_counter = 0

    task_num = 0

    while True:
        KEY_10_state = KEY_10.read()
        if KEY_10_state:
            KEY_10_counter += 1
        else:
            KEY_10_counter -= 1
        if KEY_10_counter < 0:
            KEY_10_counter = 0
        if KEY_10_counter > 30:
            KEY_10_counter = 0
            task_num = (task_num + 1) % 3
            time.sleep(1)
        

        ret, frame = cap.read()
        if not ret:
            continue

        frame = frame[180:540, 320:960]
        
        frame_drawn = frame.copy()

        colors = []
        rects = []

        if task_num == 0:
            colors, huidu_img_color = find_colors(frame_drawn, lower_color, upper_color, [20, 3000])
            rects, huidu_img_rect = find_rects(frame_drawn, lower_rect, upper_rect, [2000, 2000000])
            frame_drawn = cv2.bitwise_or(huidu_img_color, huidu_img_rect)
        elif task_num == 1:
            colors, huidu_img_color = find_colors(frame_drawn, lower_color, upper_color, [20, 3000])
            frame_drawn = huidu_img_color
        elif task_num == 2:
            rects, huidu_img_rect = find_rects(frame_drawn, lower_rect, upper_rect, [2000, 2000000])
            frame_drawn = huidu_img_rect
        for rect in rects:
            frame = my_draw_rect(frame, rect)
        for rect in colors:
            cv2.rectangle(frame, rect[0], rect[1], (0, 255, 0), 2)

        cv2.putText(frame, "fps: {}".format(round(fps, 2)), [50, 100], cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 4)
        streamer.update_frame(0, frame)
        streamer.update_frame(1, frame_drawn)
        curr_time = time.time()
        fps = (1 / (curr_time - last_time)) * 0.3 + fps * 0.7
        last_time = curr_time
        

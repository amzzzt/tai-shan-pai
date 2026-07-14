import cv2
from xbhdcc_tools import detect_cameras,WebStreamer
import time
import threading
import numpy as np
import mediapipe as mp

#MediaPipe手部21个关键点编号：0=手腕 1-4=大拇指 5-8=食指 9-12=中指 13-16=无名指 17-20=小指

#手指骨骼：四根手指每根3节
FINGER_BONES = [
    (5, 6), (6, 7), (7, 8),      #食指 3节
    (9, 10), (10, 11), (11, 12), #中指 3节
    (13, 14), (14, 15), (15, 16),#无名指 3节
    (17, 18), (18, 19), (19, 20),#小指 3节
]

#大拇指只有2节（3节要求的例外）
THUMB_BONES = [(2, 3), (3, 4)]

#手掌3节：手腕->拇指根、手腕->食指根、食指根->小指根
PALM_BONES = [(0, 2), (0, 5), (5, 17)]

ALL_BONES = PALM_BONES + THUMB_BONES + FINGER_BONES
#骨骼上用到的关节点（画圆点用）
JOINTS = sorted({i for bone in ALL_BONES for i in bone})

#===== 后台推理线程共享状态 =====
#推理耗时约210ms，放主循环里帧率只有几帧；
#丢到后台线程后主循环只画图，帧率恢复到采集速度
latest_roi = None      #主循环写入最新的中央区域画面
cached_pts = None      #推理线程写入最新的关键点（原图坐标）
state_lock = threading.Lock()
running = True

def inference_worker(hands, roi_offset, roi_size):
    """后台推理线程：不断取最新中央画面跑模型，更新关键点"""
    global cached_pts
    x0, y0 = roi_offset
    rw, rh = roi_size
    lost_count = 0
    while running:
        with state_lock:
            roi = None if latest_roi is None else latest_roi.copy()
        if roi is None:
            time.sleep(0.01)
            continue

        result = hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            #归一化坐标映射回ROI再平移到原图
            pts = [(x0 + int(lm.x * rw), y0 + int(lm.y * rh))
                   for lm in hand_lm.landmark]
            with state_lock:
                cached_pts = pts
            lost_count = 0
        else:
            lost_count += 1
            if lost_count >= 3:
                with state_lock:
                    cached_pts = None

if __name__ == "__main__":
    detect_cameras()
    #BGR
    cap = cv2.VideoCapture(9, cv2.CAP_V4L2)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC,fourcc)

    #降低分辨率换帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,60)

    streamer = WebStreamer(port=8080)
    fps = 0

    #手部识别模型：complexity=0为轻量版
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    #读一帧确定实际分辨率，算出中央识别区域
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    #只识别画面中央区域（宽高各一半），角落的手不识别
    x0, y0 = w // 4, h // 4
    x1, y1 = w * 3 // 4, h * 3 // 4

    worker = threading.Thread(target=inference_worker,
                              args=(hands, (x0, y0), (x1 - x0, y1 - y0)),
                              daemon=True)
    worker.start()

    last_time = time.time()

    while True:
        ret ,frame = cap.read()
        if not ret :
            continue

        #把最新中央画面交给推理线程（copy避免和绘制互相干扰）
        with state_lock:
            latest_roi = frame[y0:y1, x0:x1].copy()
            pts = cached_pts

        #画出识别区域边界（细绿框），手放进框内才会识别
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)

        if pts is not None:
            #红色线段拼接骨骼
            for a, b in ALL_BONES:
                cv2.line(frame, pts[a], pts[b], (0, 0, 255), 2)
            #关节点画小红点
            for i in JOINTS:
                cv2.circle(frame, pts[i], 3, (0, 0, 255), -1)

        cv2.putText(frame,"fps: {}".format(round(fps,2)),[30,30],cv2.FONT_HERSHEY_SIMPLEX,1,[255,0,0],2)

        #只推一路视频
        streamer.update_frame(0,frame)
        curr_time = time.time()
        fps = (1 / (curr_time - last_time) *0.3 + fps * 0.7)
        last_time = curr_time

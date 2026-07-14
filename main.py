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

def convert_lab_thresholds(openmv_thresholds):
    """OpenMV风格LAB阈值 [Lmin,Lmax,amin,amax,bmin,bmax] 转OpenCV范围"""
    l_min, l_max, a_min, a_max, b_min, b_max = openmv_thresholds
    lower_bound = np.array([int(l_min * 2.55), int(a_min + 128), int(b_min + 128)])
    upper_bound = np.array([int(l_max * 2.55), int(a_max + 128), int(b_max + 128)])
    return lower_bound, upper_bound

#===== 绿色花露水识别配置 =====
#绿花露水→蓝框
#L上限85：过曝的灯光亮度接近100，直接排除
#A上限-15：要求足够"绿"（防灯主力是饱和度校验，这里不用卡太死）
GREEN_HLS_LOWER, GREEN_HLS_UPPER = convert_lab_thresholds([15, 85, -128, -15, -30, 70])
GREEN_HLS_BOX_COLOR = (255, 0, 0)   #蓝色(BGR)
KERNEL_OPEN = np.ones((3, 3), np.uint8)    #小核去噪，保住小碎块
KERNEL_CLOSE = np.ones((25, 25), np.uint8) #大核闭运算：把被反光/标签/手指撕碎的瓶身粘回一块

def detect_green_bottle(frame):
    """在半分辨率图上找绿色花露水，返回原图坐标 (x,y,w,h) 或 None

    瓶身会被高光、标签、手指分割成碎块，所以：
    1.大核闭运算粘合碎块 2.所有合格碎块取并集框，而不是只挑最大块
    防灯光靠颜色阈值(亮度上限+绿度)和饱和度校验，不靠几何形状
    """
    half = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    lab = cv2.cvtColor(half, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, GREEN_HLS_LOWER, GREEN_HLS_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    hsv = None  #用到时再转，省时间
    #收集所有通过校验的绿色块，最后取并集框
    min_x = min_y = 10**9
    max_x = max_y = -1
    total_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:              #门槛放低：被遮挡的碎块本来就不大
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        #饱和度校验：过曝灯光接近白色S极低，绿瓶子S高
        if hsv is None:
            hsv = cv2.cvtColor(half, cv2.COLOR_BGR2HSV)
        region_mask = mask[y:y + bh, x:x + bw]
        mean_s = cv2.mean(hsv[y:y + bh, x:x + bw], mask=region_mask)[1]
        if mean_s < 60:
            continue
        total_area += area
        min_x = min(min_x, x); min_y = min(min_y, y)
        max_x = max(max_x, x + bw); max_y = max(max_y, y + bh)

    #所有碎块加起来还是太小，认为画面里没有瓶子
    if total_area < 400:
        return None
    #半分辨率坐标放大回原图
    return (min_x * 2, min_y * 2, (max_x - min_x) * 2, (max_y - min_y) * 2)

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

    #瓶子状态保持：检测到后维持30帧（约半秒），
    #手握瓶子时个别帧检测失败也不会闪回手掌模式
    BOTTLE_HOLD_FRAMES = 30
    bottle_box = None
    bottle_ttl = 0

    while True:
        ret ,frame = cap.read()
        if not ret :
            continue

        #先找绿色花露水：出现瓶子时只框瓶子，不标手掌
        det = detect_green_bottle(frame)
        if det is not None:
            bottle_box = det
            bottle_ttl = BOTTLE_HOLD_FRAMES
        elif bottle_ttl > 0:
            bottle_ttl -= 1
            if bottle_ttl == 0:
                bottle_box = None

        if bottle_box is not None:
            #瓶子模式：让推理线程歇着，并清掉旧骨骼，避免残影
            with state_lock:
                latest_roi = None
                cached_pts = None
                pts = None
            bx, by, bw, bh = bottle_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), GREEN_HLS_BOX_COLOR, 2)
            cv2.putText(frame, "six god", (bx, max(by - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN_HLS_BOX_COLOR, 2)
        else:
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

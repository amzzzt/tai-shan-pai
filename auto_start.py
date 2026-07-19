#!/usr/bin/env python3
"""
auto_start.py - 上电自启: 摄像头检测白矩形 → 串口发 dx/dy 给单片机
不碰 main.py, 不占 8080 端口, 不含 WebStreamer。
"""
import cv2
import time
import os
import signal
import sys
import fcntl

from white_rect_detector import WhiteRectDetector
from serial_comm import SerialComm

LOCK_FILE = '/tmp/auto_start.lock'
VIDEO_DEV = 9
SERIAL_PORT = '/dev/ttyS7'
MAX_CAMERA_RETRIES = 20
FRAME_TIMEOUT_SEC = 5  # 超过这个时间没拿到帧就认为 UVC 卡死

# ── 单实例锁 ──────────────────────────────────────
_lock_fd = open(LOCK_FILE, 'w')
try:
    fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('[auto_start] 已有实例在运行，退出。')
    sys.exit(0)

# ── 清理函数 ──────────────────────────────────────
_cap = None
_serial = None

def cleanup():
    global _cap, _serial
    print('[auto_start] 正在清理资源...')
    if _cap is not None:
        try:
            _cap.release()
        except:
            pass
        _cap = None
    if _serial is not None:
        try:
            _serial.close()
        except:
            pass
        _serial = None
    # 不杀其他进程, 只释放自己持有的资源
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_UN)
        _lock_fd.close()
    except:
        pass
    print('[auto_start] 清理完成。')

def signal_handler(sig, frame):
    print('[auto_start] 收到信号 %d，退出。' % sig)
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ── 等待摄像头就绪 ────────────────────────────────
print('[auto_start] 等待 /dev/video%d ...' % VIDEO_DEV)
for i in range(30):
    if os.path.exists('/dev/video%d' % VIDEO_DEV):
        print('[auto_start] 摄像头就绪 (第 %d 秒)' % i)
        break
    time.sleep(1)
else:
    print('[auto_start] 超时: 摄像头未就绪，退出。')
    sys.exit(1)

# ── 打开摄像头（带重试） ──────────────────────────
for attempt in range(1, MAX_CAMERA_RETRIES + 1):
    _cap = cv2.VideoCapture(VIDEO_DEV, cv2.CAP_V4L2)
    if _cap.isOpened():
        break
    print('[auto_start] 摄像头打开失败 (第 %d/%d 次)，重试...' % (attempt, MAX_CAMERA_RETRIES))
    if _cap is not None:
        _cap.release()
        _cap = None
    time.sleep(1)
else:
    print('[auto_start] 无法打开摄像头，退出。')
    sys.exit(1)

# 配置摄像头
_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
_cap.set(cv2.CAP_PROP_FPS, 30)

# ── 初始化检测器和串口 ────────────────────────────
rect = WhiteRectDetector()
try:
    _serial = SerialComm(port=SERIAL_PORT, baudrate=115200)
except Exception as e:
    print('[auto_start] 串口打开失败: %s，退出。' % e)
    cleanup()
    sys.exit(1)

print('[auto_start] 启动完成，开始主循环。')

# ── 主循环 ────────────────────────────────────────
consecutive_failures = 0
MAX_FAILURES = 60  # 连续 60 次读不到帧 ≈ 2 秒, 认为 UVC 卡死

try:
    while True:
        ret, frame = _cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= MAX_FAILURES:
                print('[auto_start] UVC 管线疑似卡死 (连续 %d 帧读取失败)，退出。' % consecutive_failures)
                break
            time.sleep(0.01)
            continue
        consecutive_failures = 0

        frame = cv2.flip(frame, 0)
        rect.detect(frame)
        _serial.send_error(rect.dx, rect.dy, rect.found)

except KeyboardInterrupt:
    pass
finally:
    cleanup()

print('[auto_start] 退出。')

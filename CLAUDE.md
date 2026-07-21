# CLAUDE.md — 回字靶视觉追踪项目 全量记忆

**一句话概述**：RK3566 摄像头拍摄回字标靶 → 检测白色内矩形 → 数学推算外框/圆心 → 单点匀速走第3圈（30秒一圈）→ 串口发偏差给云台单片机。**只检测白矩形，其余全部是算出来的。**

---

## ⚠️ 铁律（违反这些会导致硬件不可逆损坏）

1. **绝对不能碰 `/dev/video9` 的驱动/参数/权限**。摄像头只有这一个可用，没有刷机程序，堵了就死了。
2. **串口二进制帧格式绝对不能改**。`0xAA 0xBB + found(1B) + dx(2B signed short LE) + dy(2B signed short LE) + XOR(1B)`，单片机只认这个。
3. **不要新建启停脚本**。现有流程稳定运行时不要动，之前乱搞把摄像头弄堵了，只能刷机才救回来。
4. **老模块（WhiteRectDetector, BlackRectDetector, DiagRectDetector, TargetDetector, GreenBallDetector, ScanController）全部保留，不动它们。**

---

## 硬件环境

| 项目 | 值 |
|------|-----|
| 开发板 | 泰山派 RK3566 (aarch64), Ubuntu 20.04, Python 3.8.10 |
| 摄像头 | **编号 9** (`cv2.VideoCapture(9, cv2.CAP_V4L2)`)，0~8 是 rkisp 虚拟节点不能用于拍照 |
| 摄像头参数 | 640×480 MJPG, 30fps, **物理倒装**（代码 `cv2.flip(frame, 0)` 绕x轴翻转） |
| 串口 | `/dev/ttyS7`, 115200 baud |
| 预览 | `xbhdcc_tools.WebStreamer` 推 MJPEG 到 8080 端口，浏览器看 |
| 开发方式 | VS Code Remote-SSH → 泰山派 |

---

## 标靶物理尺寸（精准值）

```
白色内矩形: 25.6 × 16.9 cm  (长宽比 = 1.515)
黑色外框:   28.9 × 20.8 cm  (长宽比 = 1.389)
黑线宽度:   约 1.65cm (水平), 1.95cm (垂直) — 上下左右不均匀
5个同心圆:  圆心 = 白色矩形中心
            半径: 2, 4, 6, 8, 10 cm (每次 +2cm)
            圆是物理印在板上的图案，不参与检测，只参与追踪
```

### 关键比例（代码中硬编码的值，已经跑得很好，不要改）

| 参数 | 值 | 用途 |
|------|-----|------|
| `scale = 1.17` | 外框/内框缩放 | 从内矩形推算外框位置 |
| `radius_ratio = 6.0/17.5 = 0.343` | 第3圈半径/内框短边 | 从内矩形推算第3圈半径 |
| `ar` 理想值 `1.46` | 长宽比评分中心 | 介于内框 1.515 和外框 1.389 之间 |
| 第3圈 = 30秒一圈 | 比赛要求 | 激光绕一圈 30 秒 |

---

## 检测策略（核心设计决策）

**只检测白色内矩形。外框、中心点、5个圆的位置全部通过精准比例从白色矩形推算。**

为什么：
- 标靶移到强光源下会发生亮暗跳变（Otsu 阈值剧烈变化，白色矩形在 mask 中可能变成白可能变成黑）
- 只锁定一个可靠特征（白色矩形）比同时检测多个特征更稳定
- Otsu 正反两路 (`tval` 和 `255-tval`) 确保不管矩形是亮底还是暗底都能抓到

---

## 模块地图（只列关键文件）

| 文件 | 状态 | 作用 |
|------|------|------|
| `main.py` | **主入口** | 摄像头→V2检测→圆追踪→串口发送→Web推流 |
| `rect_detector_v2.py` | **核心检测器** | Otsu正反两路 + DP多边形逼近 + 对角线几何验证 + 填充验证(_check_fill) + 五项打分 + EMA平滑 + 追踪锁 |
| `circle_tracker.py` | **圆形路径追踪** | 时间驱动(30秒/72点)，丢框>15帧返回None暂停发数据，重新识别时用新位置+当前时间无缝接上 |
| `serial_comm.py` | **串口通信** | 二进制帧打包发送，不打印人类可读日志 |
| `xbhdcc_tools.py` | **工具库** | WebStreamer(双路MJPEG推流) + detect_cameras() |
| `auto_start.py` | 上电自启 | 独立入口，用 WhiteRectDetector |
| `white_rect_detector.py` | 保留 | 白色矩形检测（旧版，固定LAB阈值） |
| `black_rect_detector.py` | 保留 | 黑色矩形检测（内缩比例扫描） |
| `diag_rect_detector.py` | 保留 | 对角线法检测（V2原型） |
| `target_detector.py` | 保留 | Canny边缘检测（早期实验） |
| `green_ball_detector.py` | 保留 | 绿色网球检测（另一个已完善题目） |
| `scan_controller.py` | 保留 | 矩形框四角扫描（另一个题目） |
| `test_serial_rx.py` | 保留 | 串口接收测试 |

---

## main.py 执行流程（严谨版）

```
1. main.py 启动 → 杀 8080/tcp 和 /dev/video9 残留
2. 打开摄像头 /dev/video9, 640×480 MJPG 30fps
3. 初始化: WebStreamer(8080), RectDetectorV2, CircleTracker(rd), SerialComm(ttyS7)
4. 主循环:
   a. cap.read() → flip(frame, 0) 翻转
   b. rd.detect(frame) 检测白色矩形
   c. confirm 计数器: rd.found=True → +1, False → 归零. confirmed = confirm >= 2
   d. 确认后启动圆追踪(只一次): ct.start(period=30.0, n_points=72)
   e. ct.update(h, w) 获取当前追踪点 → 丢框≤15帧照常返回, >15帧返回None
   f. 串口发送: ct_pt非None → send_error(dx, dy, True), None → send_error(0, 0, False)
   g. 绘制: 中心十字, 矩形框(绿色, 需confirmed), 圆路径(青色点), 追踪红点, FPS
   h. 双路推流: channel0=绘制后画面, channel1=检测mask
```

---

## CircleTracker 状态机

```
┌──────────┐  start()   ┌──────────┐  丢框>15帧   ┌────────────┐
│  inactive │ ────────→ │  active   │ ────────────→ │ active但返  │
│ points=[] │           │ 正常返点  │               │ 回None     │
└──────────┘           └──────────┘              └────────────┘
                              ↑                        │
                              │  重新识别到              │
                              └────────────────────────┘
                              
时间不停、路径保留、active不变。
重新识别时: _lost_frames=0 → _generate_path()用新位置 → 当前时间接上。
```

**封闭圆设计**：72个点均匀分布，秒数取模后 progress 从 99%→0% 时，首尾相邻点距极短，云台无感。

---

## RectDetectorV2 检测链路

```
frame → gray → GaussianBlur(5,5) → Otsu → tval
                                          ↓
                              for tv in [tval, 255-tval]:
                                mask = threshold(gray, tv)
                                OPEN(5,5) + CLOSE(5,5)
                                _find_quads_dp(mask, img_area)
                                  ├─ 面积过滤 (img_area/300 ~ img_area/2)
                                  ├─ 凸性 ≥ 0.80
                                  ├─ DP逼近 (0.04*peri) + 严格版(0.02*peri) 
                                  ├─ 3点补角 / 4点通收 / 5~8点回退minAreaRect
                                  └─ 返回 (corners, area, area_ratio, peri_ratio, n_strict)
                                ↓
                              筛选 top-8 面积 → 逐一评分:
                                ├─ 边界排除 (距画面边缘<5px拒)
                                ├─ n_strict > 6 拒 (弯边炸边)
                                ├─ peri_ratio > 1.30 拒
                                ├─ area_ratio < 0.65 拒
                                ├─ is_valid_rect() (对角线15% + 中点8% + 邻边12°)
                                ├─ _check_fill() (内部白 + 外环暗 + ≥2条暗边)
                                ├─ 角度质量 ≥ 0.25
                                └─ 五项打分 → best_score
                                ↓
                              first found → break (Otsu两路不用跑完)
                              ↓
              最低分门槛: locked? 0.35 : 0.45
              亚像素精化 cornerSubPix(3,3)
              角点排序 [TL,TR,BR,BL]
              EMA平滑 new*0.8+old*0.2
              追踪锁: 位移<0.8×框尺寸 & 面积变化<40% & 长宽比变化<0.30
```

### _check_fill 逻辑（区分标靶和白墙的关键）
```
1. 矩形内部 mean(gray) > 35 (够亮)
2. 矩形外 ring (max(3, min(14, max(w,h)*0.25)) 宽度) 的 mean < 内部×0.78
3. 至少 2~3 条边暗 → 有黑边框 → 是标靶
```

---

## 串口二进制帧（绝对不能改）

```
字节0-1: 0xAA 0xBB (帧头)
字节2:   found (0x01=有目标, 0x00=无目标)
字节3-4: dx (signed short, little-endian) — 目标点x - 画面中心x
字节5-6: dy (signed short, little-endian) — 目标点y - 画面中心y
字节7:   XOR校验 (字节2-6逐字节异或)
```

---

## 已知问题 & 不做的事

1. **亮暗跳变**：强光源下 Otsu 阈值跳变，可能短暂丢框。15帧容忍窗口+追踪锁缓冲，当前够用。
2. **圆路径抖动**：每帧微调路径点位置（因为检测框抖动），系统完善后再精细处理。
3. **丢框后绘制**：1~15帧丢框时矩形框/圆路径消失但追踪红点还在走。用户确认保持现状。
4. **不要新建启停脚本**。之前乱搞把摄像头弄堵了，现有流程稳定。
5. **不要改串口格式**。
6. **不要删任何保留模块**。

---

## 记忆文件索引

- `memory/taishanpai-project-env.md` — 硬件环境详情
- `memory/target-dimensions.md` — 标靶精准尺寸和比例
- `memory/MEMORY.md` — 总索引

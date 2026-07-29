"""
GPIO 按键模块 — 防抖 + 单击检测

用法:
  from gpio_button import GpioButton
  btn = GpioButton("GPIO3_A1")   # 传字符串
  btn = GpioButton(97)           # 或传数字

  while True:
      if btn.update():           # 每帧调一次，返回 True 表示按了一下
          print("按了一下")
"""

from periphery import GPIO


class GpioButton:
    def __init__(self, pin, threshold=30):
        """
        pin:      字符串如 "GPIO3_A1"，或 Linux GPIO 编号如 97
        threshold: 消抖，连续按住多少帧触发（默认 30）
        """
        if isinstance(pin, str):
            pin = self._name_to_num(pin)

        self._gpio = GPIO(pin, "in")
        self._threshold = threshold
        self._counter = 0

    @staticmethod
    def _name_to_num(name):
        """GPIO3_B5 → 109, GPIO3_A1 → 97"""
        # name 格式: "GPIO{组号}_{字母}{尾号}"
        base = name[4:]            # "3_B5"
        group_str, rest = base.split('_')  # "3", "B5"
        group = int(group_str)
        letter = rest[0]           # "B"
        num = int(rest[1:])        # "5"
        offset = ord(letter) - ord('A')
        return group * 32 + offset * 8 + num

    def update(self):
        """每帧调用，检测到有效按键返回 True"""
        if self._gpio.read():
            self._counter += 1
        else:
            self._counter -= 1

        if self._counter < 0:
            self._counter = 0

        if self._counter > self._threshold:
            self._counter = 0
            return True
        return False

    def raw(self):
        """返回 GPIO 当前电平，调试用"""
        return self._gpio.read()

    def close(self):
        self._gpio.close()

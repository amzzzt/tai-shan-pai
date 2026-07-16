"""串口通信模块 — 二进制帧发送"""
import struct
import time
import serial


class SerialComm:
    def __init__(self, port='/dev/ttyS7', baudrate=115200):
        self.ser = serial.Serial(port, baudrate)
        self._last_send = 0.0

    def send_error(self, dx, dy, found):
        now = time.time()
        if now - self._last_send < 0.05:
            return
        self._last_send = now
        try:
            found_byte = 0x01 if found else 0x00
            payload = struct.pack('<Bhh', found_byte, int(dx), int(dy))
            chk = 0
            for b in payload:
                chk ^= b
            self.ser.write(b'\xAA\xBB' + payload + bytes([chk & 0xFF]))
        except Exception:
            pass

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

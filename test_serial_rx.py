import serial
import time

ser = serial.Serial('/dev/ttyS7', 115200, timeout=0.1)
print('[RX] Listening on ttyS7...')

while True:
    if ser.in_waiting > 0:
        data = ser.read(ser.in_waiting)
        print('[RX] got %d bytes: %s' % (len(data), data.hex()))
    time.sleep(0.1)

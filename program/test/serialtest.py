import serial
import time

ser = serial.Serial('/dev/ttyUSB0', 115200)
time.sleep(2)
with ser:
    ser.write(b'0:0')
    print(ser.readline().decode('utf-8').strip())
    print(ser.readline().decode('utf-8').strip())
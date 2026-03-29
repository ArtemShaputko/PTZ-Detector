import threading
import time
import serial
from utils import get_distance, is_in_ellipse
from logger import Logger

class SerialWriter:
    def __init__(self, logger: Logger | None = None, size = (1920, 1080), notsend_zone_factor = 0.04):
        self.__ser = serial.Serial('/dev/ttyUSB0', 115200)
        self.__size = size
        self.__screen_center = (size[0] // 2, size[1] // 2)
        self.__not_send_zone = (self.__size[0] * notsend_zone_factor, self.__size[1] * notsend_zone_factor)
        self.__notsend_zone_factor = notsend_zone_factor
        
        self.__logger = logger
        
        self.__lock = threading.Lock()
        self.__coords = None
        self.__stop = threading.Event()
    
    @property 
    def not_send_zone(self):
        return self.__not_send_zone
    
    @not_send_zone.setter
    def not_send_zone(self, zone):
        with self.__lock:
            self.__not_send_zone = zone
            
    def update_notsend_zone_by_size(self, size):
        with self.__lock:
            self.__size = size
            self.__screen_center = (size[0] // 2, size[1] // 2)
            self.__not_send_zone = (self.__size[0] * self.__notsend_zone_factor, self.__size[1] * self.__notsend_zone_factor)
        
    def stop(self):
        self.__stop.set()
    
    @property
    def coords(self):
        with self.__lock:
            return self.__coords
        
    @coords.setter
    def coords(self, coords):
        with self.__lock:
            self.__coords = coords
    
    def write_loop(self):
        if self.__logger:
            self.__logger.info('SerialWriter Loop started')
        
        while not self.__stop.is_set():
            data_to_send = None
            
            with self.__lock:
                current_val = self.__coords
                self.__coords = None
                
            if current_val is not None:
                a = self.__not_send_zone[0] / 2
                b = self.__not_send_zone[1] / 2
                if not is_in_ellipse(self.__screen_center, current_val, a, b):
                    data_to_send = f'{current_val[0]}:{current_val[1]}\n'
                else:
                    data_to_send = f'{self.__screen_center[0]}:{self.__screen_center[1]}\n'
                    
            
            if data_to_send:
                try:
                    if self.__logger:
                        self.__logger.trace(f'Coords to send: {current_val}')
                    self.__ser.reset_input_buffer()
                    self.__ser.write(data_to_send.encode('utf-8'))
                    self.__ser.flush()
                    time.sleep(0.025) 
                except Exception as e:
                    if self.__logger:
                        self.__logger.error(f"Serial Write Error: {e}")
                    break
            
            time.sleep(0.005)
            
        self.__ser.close()
        if self.__logger:
            self.__logger.info("SerialWriter stopped")
        
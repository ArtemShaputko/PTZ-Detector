import threading
import time
import serial
from utils import get_distance
from logs import Logger

class SerialWriter:
    def __init__(self, logger: Logger | None = None, min_distance_to_send = 30):
        self.__ser = serial.Serial('/dev/ttyUSB0', 115200)
        self.__last_sent = None
        self.__min_distance_to_send = min_distance_to_send
        
        self.__logger = logger
        
        self.__coords_lock = threading.Lock()
        self.__coords = None
        self.__stop = threading.Event()
        
    def stop(self):
        self.__stop.set();
        
    def set_coords(self, coords):
        with self.__coords_lock:
            self.__coords = coords
    
    def write_loop(self):
        if self.__logger:
            self.__logger.info('SerialWriter Loop started')
        
        while not self.__stop.is_set():
            data_to_send = None
            
            with self.__coords_lock:
                current_val = self.__coords
                if current_val is not None:
                    if self.__last_sent is None or get_distance(current_val, self.__last_sent) > self.__min_distance_to_send:
                        data_to_send = f'{current_val[0]}:{current_val[1]}\n'
                        self.__last_sent = current_val
            
            if data_to_send:
                try:
                    self.__ser.reset_input_buffer()
                    self.__ser.write(data_to_send.encode('utf-8'))
                    self.__ser.flush()
                    time.sleep(0.01) 
                except Exception as e:
                    if self.__logger:
                        self.__logger.error(f"Serial Write Error: {e}")
                    break
            
            time.sleep(0.005)
            
        self.__ser.close()
        if self.__logger:
            self.__logger.info("SerialWriter stopped")
        
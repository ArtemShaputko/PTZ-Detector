import threading
import cv2
from logs import Logger

class ZoomController:
    def __init__(self, logger: Logger | None = None, min_zoom=1.0, max_zoom=5.0, step=0.5):
        self.__zoom = 1.0
        self.__min = min_zoom
        self.__max = max_zoom
        self.__step = step
        self.__lock = threading.Lock()
        
        self.__logger = logger

    def zoom_in(self):
        with self.__lock:
            self.__zoom = min(self.__zoom + self.__step, self.__max)
            if self.__logger:
                self.__logger.info(f"[Zoom] приблизить → x{self.__zoom:.1f}")

    def zoom_out(self):
        with self.__lock:
            self.__zoom = max(self.__zoom - self.__step, self.__min)
            if self.__logger:
                self.__logger.info(f"[Zoom] удалить → x{self.__zoom:.1f}")

    def get_zoom(self):
        with self.__lock:
            return self.__zoom

    def apply(self, frame):
        zoom = self.get_zoom()
        if zoom == 1.0:
            return frame

        h, w = frame.shape[:2]
        new_h, new_w = int(h / zoom), int(w / zoom)

        y1 = (h - new_h) // 2
        x1 = (w - new_w) // 2
        cropped = frame[y1:y1 + new_h, x1:x1 + new_w]

        return cv2.resize(cropped, (w, h))
    
    def to_original_coords(self, cx: int, cy: int, orig_w: int, orig_h: int) -> tuple[int, int]:
        zoom = self.get_zoom()
        if zoom == 1.0:
            return cx, cy

        crop_w, crop_h = int(orig_w / zoom), int(orig_h / zoom)

        offset_x = (orig_w - crop_w) // 2
        offset_y = (orig_h - crop_h) // 2

        orig_cx = int(cx / zoom) + offset_x
        orig_cy = int(cy / zoom) + offset_y
        return orig_cx, orig_cy
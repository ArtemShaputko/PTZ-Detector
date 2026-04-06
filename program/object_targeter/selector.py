import torch
import threading
from ultralytics.engine.results import Results, Boxes

class ObjectSelector:
    def __init__(self):
        self.__lock = threading.Lock
        self.__target_class: int | None = None
        
    @property
    def target_class(self):
        with self.__lock:
            return self.__target_class
        
    @staticmethod
    def is_target(boxes: Boxes, target: int | None) -> bool:
        return boxes.is_track and target is not None and boxes.id == target
        
    def select_best(self, results: Results, target: int | None) -> tuple[int, int] | None:
        best_box = None
        best_conf = -1
        
        result = results[0]

        for i in range(len(result)):
            if self.is_target(result.boxes, target):
                best_box = result.boxes.xyxy[i]
                break
            conf = result.boxes.conf[i].item()
            if conf > best_conf:
                best_conf = conf
                best_box = result.boxes.xyxy[i]

        if best_box is None:
            return None

        x1, y1, x2, y2 = best_box.to(torch.int).tolist()
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return cx, cy
    
    def select_first(self, results) -> tuple[int, int] | None:
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        best_box = results[0].boxes.xyxy[0]
        x1, y1, x2, y2 = best_box.to(torch.int).tolist()
        return (x1 + x2) // 2, (y1 + y2) // 2

    def select(self, results, type='first') -> tuple[int, int] | None:
        coords = self.select_best(results) if type == 'best' else self.select_first(results)
        if coords is None:
            return None
        return coords
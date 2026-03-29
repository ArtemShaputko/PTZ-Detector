import torch
from zoom import ZoomController

class ObjectSelector:
    def __init__(self, width: int, height: int, zoom: ZoomController):
        self.__width = width
        self.__height = height
        self.__zoom = zoom
        
    def select_best(self, results) -> tuple[int, int] | None:
        best_box = None
        best_conf = -1

        for result in results:
            for i in range(len(result)):
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
        cx, cy = self.__zoom.to_original_coords(coords[0], coords[1])
        return (self.__width - cx, cy)
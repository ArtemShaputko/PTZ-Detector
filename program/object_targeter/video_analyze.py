import cv2
from ultralytics import YOLOWorld
import torch
import time
from names import Names
from serialwriter import SerialWriter
from selector import ObjectSelector
from zoom import ZoomController
from preprocessor import Preprocessor
from logs import Logger
from smooth import SmoothingFilter
import numpy as np
import tkinter as tk

class Overlay:
    __NORMAL_THICKNESS = 2
    __TARGET_THICKNESS = 4
    __FONT = cv2.FONT_HERSHEY_COMPLEX
    __FONT_SCALE = 0.7
    __SMALL_FONT_SCALE = 0.6

    def draw(self, frame, results, classes_names: dict, zoom_level: float,
             target_idx: int = 0, colors_fn=None):
        if not results or results[0].boxes is None:
            return frame

        boxes = results[0].boxes
        keys = list(classes_names.keys())
        
        # Уровень зума — сверху слева
        cv2.putText(frame, f"Zoom: x{zoom_level:.1f}", (10, 25),
                self.__FONT, self.__SMALL_FONT_SCALE, (255, 255, 0), 1)

        h = frame.shape[0]
        for i, name in enumerate(keys):
            cv2.putText(frame, name, (10, h - 10 - i * 15),
                        self.__FONT, self.__SMALL_FONT_SCALE,
                        (255, 255, 255), 1)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].to(torch.int).tolist()
            conf = boxes.conf[i].item()
            cls  = int(boxes.cls[i].item())

            color     = colors_fn(cls) if colors_fn else (0, 255, 0)
            thickness = self.__TARGET_THICKNESS if i == target_idx \
                        else self.__NORMAL_THICKNESS

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{keys[cls]}, conf={conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        self.__FONT, self.__FONT_SCALE, color, thickness)

        return frame

    def fit_to_screen(self, frame, screen_w: int, screen_h: int):
        fh, fw = frame.shape[:2]
        scale = min(screen_w / fw, screen_h / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        y = (screen_h - new_h) // 2
        x = (screen_w - new_w) // 2
        canvas[y:y+new_h, x:x+new_w] = resized
        return canvas


class VideoAnalyzer:
    def __init__(self,
                 names: Names,
                 zoom: ZoomController,
                 serial_writer: SerialWriter,
                 preprocessor: Preprocessor | None = None,
                 logger: Logger | None = None,
                 model_name = "yolov8m-worldv2.pt",
                 size = (1920, 1080), conf_score = 0.1):
        self.__names = names
        self.__serial_writer = serial_writer
        self.__selector = ObjectSelector(size[0], size[1], zoom)
        self.__zoom = zoom
        self.__preprocessor = preprocessor
        self.__smoother = SmoothingFilter(window=2)
        self.__overlay = Overlay()
        self.__logger = logger
        self.__to_work = True

        self.__model = YOLOWorld(model_name)
        self.__model.to('cuda')

        self.__conf_score = conf_score
        self.__size = size

        # Размер экрана
        root = tk.Tk()
        self.__screen_w = root.winfo_screenwidth()
        self.__screen_h = root.winfo_screenheight()
        root.destroy()

        self.__cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.__cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.__cap.set(cv2.CAP_PROP_FPS, 30)
        self.__cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        res_w = self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        res_h = self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if self.__logger:
            self.__logger.info(f"Resolution: {res_w}x{res_h}, "
                               f"Screen: {self.__screen_w}x{self.__screen_h}")

        time.sleep(2)

    def __set_classes(self):
        changed, classes_names = self.__names.get_names()
        if changed:
            start_time = time.time()
            self.__model.model.clip_model = None
            self.__model.set_classes(classes_names.values())
            if self.__logger:
                self.__logger.info(f"Classes changed: {list(classes_names.keys())} "
                                   f"in {time.time()-start_time:.2f}s")
        return classes_names

    def start(self):
        self.__cap.read()
        total_frames = 0
        record_start_time = time.time()
        
                # Окно
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        while self.__cap.isOpened() and self.__to_work:
            success, frame = self.__cap.read()
            if not success:
                break

            classes_names = self.__set_classes()

            frame = self.__zoom.apply(frame)
            preprocessed = frame
            if self.__preprocessor:
                preprocessed = self.__preprocessor.apply(frame)

            results = self.__model.track(preprocessed, persist=True,
                                         verbose=False, conf=self.__conf_score)

            frame = self.__overlay.draw(frame, results, classes_names, self.__zoom.get_zoom(),
                                        target_idx=0,
                                        colors_fn=self.__names.colors)

            coords = self.__selector.select(results)
            coords = self.__smoother.update(coords)

            if coords:
                self.__serial_writer.set_coords(coords)
                if self.__logger:
                    self.__logger.info(f"Coords: {coords}")

            # Вписываем в экран с сохранением соотношения сторон
            display = self.__overlay.fit_to_screen(frame,
                                                   self.__screen_w,
                                                   self.__screen_h)
            cv2.imshow("Tracking", display)
            cv2.waitKey(1)

            total_frames += 1

            if not self.__names.get_to_work():
                break

        self.__serial_writer.stop()
        if self.__logger:
            self.__logger.info(f"FPS average: "
                               f"{total_frames / (time.time() - record_start_time):.1f}")
        self.__cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.__to_work = False
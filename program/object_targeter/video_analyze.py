import cv2
from ultralytics import YOLOWorld
import torch
import time
import subprocess
import numpy as np
import tkinter as tk

from names import Names
from serialwriter import SerialWriter
from selector import ObjectSelector
from zoom import ZoomController
from preprocessor import Preprocessor
from logger import Logger
from smooth import SmoothingFilter
import threading


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
            cls = int(boxes.cls[i].item())

            color = colors_fn(cls) if colors_fn else (0, 255, 0)
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
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas


class IOOperator:
    """
    Capture + display in one thread.

    Captures raw frames from FFmpeg, stores the latest raw frame
    for VideoAnalyzer, and displays zoomed frames with overlay.
    """

    def __init__(self, size: tuple[int, int], zoom: ZoomController,
                 names: Names, analyzer, logger: Logger | None):
        self.__size = size
        w, h = self.__size
        self.__zoom = zoom
        self.__analyzer = analyzer
        self.__names = names
        self.__logger = logger

        self.__chunk = w * h * 3

        # Latest RAW frame (before zoom) for VideoAnalyzer
        self.__latest_raw: np.ndarray | None = None
        self.__raw_lock = threading.Lock()

        self.__running = True
        self.__overlay = Overlay()

        root = tk.Tk()
        self.__screen_w = root.winfo_screenwidth()
        self.__screen_h = root.winfo_screenheight()
        root.destroy()

        cmd = [
            "ffmpeg", "-f", "v4l2", "-framerate", "30",
            "-video_size", f"{w}x{h}",
            "-input_format", "mjpeg",
            "-fflags", "+nobuffer+discardcorrupt",
            "-avioflags", "direct",
            "-flags", "+low_delay",
            "-thread_queue_size", "1",
            "-i", "/dev/video2",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "-"
        ]
        self.__proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        if self.__logger:
            self.__logger.info(f"Resolution: {w}x{h}, "
                               f"Screen: {self.__screen_w}x{self.__screen_h}")

        self.__thread = threading.Thread(target=self.__io_loop, daemon=True)
        self.__thread.start()

    def __io_loop(self):
        total_frames = 0
        record_start_time = time.time()
        w, h = self.__size

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        while self.__running:
            raw = self.__proc.stdout.read(self.__chunk)
            if len(raw) < self.__chunk:
                break

            # .copy() важен — np.frombuffer даёт view на буфер трубы
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()

            # Сохраняем RAW кадр для VideoAnalyzer
            with self.__raw_lock:
                self.__latest_raw = frame

            # Зумируем для отображения (на своей копии)
            display_frame = self.__zoom.apply(frame)

            model_results = self.__analyzer.get_results()
            _, class_names = self.__names.get_names(Names.CONSUMER_IO)

            display_frame = self.__overlay.draw(
                display_frame, model_results, class_names,
                self.__zoom.zoom / 10,
                target_idx=0,
                colors_fn=self.__names.colors
            )

            display_frame = self.__overlay.fit_to_screen(
                display_frame, self.__screen_w, self.__screen_h
            )
            cv2.imshow("Tracking", display_frame)
            cv2.waitKey(1)

            total_frames += 1

        self.__proc.terminate()
        self.__proc.wait()
        if self.__logger:
            elapsed = time.time() - record_start_time
            self.__logger.info(
                f"IO FPS average: {total_frames / elapsed:.1f}"
            )
        cv2.destroyAllWindows()

    def get_latest_raw(self) -> np.ndarray | None:
        """Возвращает копию последнего RAW кадра для модели."""
        with self.__raw_lock:
            return self.__latest_raw.copy() if self.__latest_raw is not None else None

    def stop(self):
        self.__running = False

        if self.__proc.stdout:
            if self.__logger:
                self.__logger.info("Closing ffmpeg stdout")
            self.__proc.stdout.close()

        try:
            if self.__logger:
                self.__logger.info("Terminating ffmpeg process")
            self.__proc.terminate()
            self.__proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if self.__logger:
                self.__logger.warning("Can't terminate ffmpeg — killing it")
            self.__proc.kill()
            self.__proc.wait()


class VideoAnalyzer:
    def __init__(self,
                 names: Names,
                 zoom: ZoomController,
                 serial_writer: SerialWriter,
                 preprocessor: Preprocessor | None = None,
                 logger: Logger | None = None,
                 model_name="yolov8s-worldv2.pt",
                 size=(1920, 1080), conf_score=0.1):
        self.__names = names
        self.__serial_writer = serial_writer
        self.__selector = ObjectSelector(size[0], size[1], zoom)
        self.__smoother = SmoothingFilter(window=2)
        self.__zoom = zoom
        self.__preprocessor = preprocessor
        self.__logger = logger
        self.__to_work = True

        self.__results_lock = threading.Lock()
        self.__model_results = None

        self.__model = YOLOWorld(model_name)
        self.__model.to('cuda')

        self.__conf_score = conf_score
        self.__size = size

        # IOOperator стартует свой поток внутри __init__
        self.__io = IOOperator(self.__size, zoom, names, self, logger=logger)

    def get_results(self):
        """Возвращает последние results модели. Без копирования — только чтение bbox."""
        with self.__results_lock:
            return self.__model_results

    def __set_classes(self):
        changed, class_names = self.__names.get_names(Names.CONSUMER_ANALYZER)
        if changed:
            start_time = time.time()
            self.__model.model.clip_model = None
            self.__model.set_classes(class_names.values())
            if self.__logger:
                self.__logger.info(
                    f"Classes changed: {list(class_names.keys())} "
                    f"in {time.time() - start_time:.2f}s"
                )
        return class_names

    def start(self):
        total_frames = 0
        record_start_time = time.time()

        while self.__to_work:
            # Берём свежий RAW кадр
            frame = self.__io.get_latest_raw()
            if frame is None:
                continue

            self.__set_classes()

            # Зумируем для модели (независимо от IO потока)
            frame = self.__zoom.apply(frame)

            preprocessed = frame
            if self.__preprocessor:
                preprocessed = self.__preprocessor.apply(frame)

            model_results = self.__model.track(
                preprocessed, persist=True,
                verbose=False, conf=self.__conf_score
            )

            with self.__results_lock:
                self.__model_results = model_results

            coords = self.__selector.select(model_results)
            coords = self.__smoother.update(coords)

            if coords:
                if self.__logger:
                    self.__logger.trace(f"Video sends coords: {coords}")
                self.__serial_writer.coords = coords

            total_frames += 1

            if not self.__names.get_to_work():
                self.stop()

        self.__serial_writer.stop()
        if self.__logger:
            elapsed = time.time() - record_start_time
            self.__logger.info(
                f"Analyzer FPS average: {total_frames / elapsed:.1f}"
            )

    def stop(self):
        self.__to_work = False
        self.__io.stop()
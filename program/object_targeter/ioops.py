
import cv2
import torch
import time
import subprocess
import numpy as np
import tkinter as tk

from config import WorkConfigManager, WorkConfig
from zoom import ZoomController
from logger import Logger
import threading
from ultralytics.engine.results import Results
from threadmanager import ThreadManager

class Overlay:
    __NORMAL_THICKNESS = 2
    __TARGET_THICKNESS = 4
    __FONT = cv2.FONT_HERSHEY_COMPLEX
    __FONT_SCALE = 0.7
    __SMALL_FONT_SCALE = 0.6

    def draw(self, frame: np.ndarray, results: list[Results], config: WorkConfig, zoom_level: float,
             target_idx: int = 0, colors_fn=None):
        if not results or results[0].boxes is None:
            return frame
        
        boxes = results[0].boxes
        fh, fw, _ = frame.shape
        oh, ow = results[0].orig_shape
        
        scale_y, scale_x = fh/oh, fw/ow
        keys = list(config.names.keys())

        cv2.putText(frame, f"Зум: x{zoom_level:.1f}", (10, 25),
                    self.__FONT, self.__SMALL_FONT_SCALE, (255, 255, 0), 1)
        cv2.putText(frame, f"Уверенность: {config.conf*100:.0f}%", (10, 50),
                    self.__FONT, self.__SMALL_FONT_SCALE, (0, 255, 255), 1)

        h = frame.shape[0]
        for i, name in enumerate(keys):
            cv2.putText(frame, name, (10, h - 10 - i * 15),
                        self.__FONT, self.__SMALL_FONT_SCALE,
                        (255, 0, 255), 1)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].to(torch.int).tolist()
            conf = boxes.conf[i].item()
            cls = int(boxes.cls[i].item())

            color = colors_fn(cls) if colors_fn else (0, 255, 0)
            thickness = self.__TARGET_THICKNESS if i == target_idx \
                else self.__NORMAL_THICKNESS
                
            x1, y1, x2, y2 = int(x1*scale_x), int(y1 * scale_y), int(x2* scale_x), int(y2*scale_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{keys[cls]}, ув={conf*100:.0f}f"
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


class IOOperator(ThreadManager):

    def __init__(self, size: tuple[int, int], zoom: ZoomController,
                 config_manager: WorkConfigManager, logger: Logger | None):
        super().__init__()
        
        self.__analyzer = None
        
        self.__size = size
        w, h = self.__size
        self.__zoom = zoom
        self.__config_manager = config_manager
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
            
    
    @property        
    def analyzer(self):
        return self.__analyzer
    
    @analyzer.setter        
    def analyzer(self, analyzer):
        self.__analyzer = analyzer

    def start(self):
        analyzer = self.__analyzer
        if not analyzer:
            raise ValueError("First should be set analyzer")
        
        total_frames = 0
        record_start_time = time.time()
        w, h = self.__size

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        while self.__running:
            raw = self.__proc.stdout.read(self.__chunk)
            if len(raw) < self.__chunk:
                if self.__logger:
                    self.__logger.error("Readed less than chunk data")
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()

            with self.__raw_lock:
                self.__latest_raw = frame

            display_frame = self.__zoom.apply(frame)

            model_results = analyzer.get_results()
            config = self.__config_manager.config
            if not config.to_work:
                self.stop()
                break

            display_frame = self.__overlay.draw(
                display_frame, model_results, config,
                self.__zoom.zoom / 10,
                target_idx=0,
                colors_fn=self.__config_manager.colors
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
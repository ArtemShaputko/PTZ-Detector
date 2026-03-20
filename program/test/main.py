#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import threading
import time
import numpy as np
import cv2
import torch
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, 
    QPushButton, QHBoxLayout, QTextEdit, QSlider, QGroupBox, 
    QMessageBox, QProgressBar, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QLinearGradient
from vosk import Model, KaldiRecognizer, SetLogLevel
import sounddevice as sd

SetLogLevel(-1)

COMMANDS_FILE = "voice_commands.json"
DETECTIONS_FILE = "detections.json"
VOSK_MODEL_PATH = "vosk-model-small-ru-0.22"
YOLO_MODEL_NAME = "yolov8m-worldv2.pt"

ru_model_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

ru_to_id = {
    'бутылка': 39, 'телефон': 67, 'чашка': 41, 'ноутбук': 63,
    'мышка': 64, 'клавиатура': 66, 'часы': 74, 'книга': 73,
    'сумка': 26, 'зонт': 25, 'нож': 43, 'вилка': 42, 'ложка': 44,
    'бутылочка': 39, 'телефончик': 67, 'чашечка': 41, 'кружка': 41,
    'мобильник': 67, 'смартфон': 67, 'компьютер': 63, 'лампа': 62,
    'мышь': 64, 'монитор': 62, 'телевизор': 62, 'экран': 62,
    'человек': 0, 'люди': 0, 'кот': 15, 'кошка': 15, 'собака': 16,
    'пёс': 16, 'машина': 2, 'авто': 2, 'автобус': 5
}


class NamesManager:
    def __init__(self):
        self.updated = False
        self.current = []
        self.current_en_names = []
        self.to_work = True
        self.lock = threading.Lock()
        self.work_lock = threading.Lock()
        self.command_history = []
        self.detecting = False
        self.last_command = ""
        
    def add(self, text):
        text = text.lower().strip()
        stop_words = ['пожалуйста', 'мне', 'хочу', 'нужно', 'давай', 'быстро', 'сейчас', 'включи', 'стоп']
        for word in stop_words:
            text = text.replace(word, '')
        text = ' '.join(text.split())
        
        if "найди" in text:
            parts = text.split("найди", 1)
            object_name = parts[1].strip() if len(parts) > 1 else ""
        else:
            object_name = text
        
        if not object_name:
            return False
        
        class_id = None
        en_name = None
        
        for ru_name, cid in ru_to_id.items():
            if ru_name in object_name or object_name in ru_name:
                class_id = cid
                en_name = ru_model_names.get(cid)
                break
        
        if class_id is None:
            for cid, name in ru_model_names.items():
                if name.lower() in object_name or object_name in name.lower():
                    class_id = cid
                    en_name = name
                    break
        
        if class_id is None:
            fuzzy_map = {
                'бутыл': 39, 'bottl': 39,
                'телеф': 67, 'phon': 67, 'mobil': 67,
                'чаш': 41, 'cup': 41, 'mug': 41, 'круж': 41,
                'ноут': 63, 'lapt': 63,
                'мыш': 64, 'mous': 64,
                'клав': 66, 'keyb': 66,
                'час': 74, 'clock': 74,
                'книг': 73, 'book': 73,
                'человек': 0, 'person': 0, 'люди': 0,
            }
            for key, cid in fuzzy_map.items():
                if key in object_name:
                    class_id = cid
                    en_name = ru_model_names.get(cid)
                    break
        
        if class_id is not None and en_name:
            with self.lock:
                self.current = [class_id]
                self.current_en_names = [en_name]
                self.updated = True
                self.detecting = True
                self.last_command = text
                cmd = {
                    "timestamp": datetime.now().isoformat(), 
                    "command": text, 
                    "object": object_name, 
                    "class_id": class_id, 
                    "en_name": en_name
                }
                self.command_history.append(cmd)
                self._save_commands()
            return True
        return False
    
    def get_status(self):
        with self.lock:
            updated = self.updated
            self.updated = False
            return updated, self.current.copy()

    def clear(self):
        with self.lock:
            self.current = []
            self.current_en_names = []
            self.updated = True
            self.detecting = False
            self.last_command = ""
    
    def get_en_names(self):
        with self.lock:
            return self.current_en_names.copy()
    
    def get_to_work(self):
        with self.work_lock:
            return self.to_work
        
    def set_to_work(self, to_work):
        with self.work_lock:
            self.to_work = to_work

    def get_last_command(self):
        with self.lock:
            return self.last_command
            
    def get_command_history(self):
        with self.lock:
            return self.command_history.copy()
            
    def _save_commands(self):
        try:
            with open(COMMANDS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.command_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save commands: {e}")
            
    def load_commands(self):
        try:
            if os.path.exists(COMMANDS_FILE):
                with open(COMMANDS_FILE, "r", encoding="utf-8") as f:
                    self.command_history = json.load(f)
        except:
            pass


class AudioLevelMeter(QWidget):
    def __init__(self):
        super().__init__()
        self.level = 0.0
        self.setFixedHeight(30)
        self.setMinimumWidth(200)
        
    def set_level(self, level):
        self.level = min(1.0, max(0.0, level))
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#1a1a2e"))
        gradient = QLinearGradient(0, 0, w, 0)
        gradient.setColorAt(0, QColor("#4ecca3"))
        gradient.setColorAt(0.5, QColor("#45a049"))
        gradient.setColorAt(1, QColor("#d20f39"))
        bar_w = int(w * self.level)
        if bar_w > 0:
            painter.fillRect(0, 2, bar_w, h-4, gradient)
        painter.setPen(QColor("#45475a"))
        painter.drawRect(0, 0, w-1, h-1)


class AudioRecorder(QThread):
    recognized = pyqtSignal(str)
    status = pyqtSignal(str)
    audio_level = pyqtSignal(float)
    debug_text = pyqtSignal(str)
    command_detected = pyqtSignal(str)
    
    def __init__(self, model_path=VOSK_MODEL_PATH, fs=16000, names_manager=None):
        super().__init__()
        self.fs = fs
        self.model_path = model_path
        self.names = names_manager
        self.running = False
        self.sensitivity = 3
        self.gain = 10.0
        self.chunk_duration = 0.2
        self.model = None
        self.rec = None
        self.callback_count = 0
        self.stream = None
        
    def set_sensitivity(self, value):
        self.sensitivity = value
        
    def set_gain(self, value):
        self.gain = float(value)
        
    def check_model(self):
        if not os.path.exists(self.model_path):
            return False, f"Folder not found: {self.model_path}"
        required = ["am", "graph", "mfcc.conf"]
        for f in required:
            if not os.path.exists(os.path.join(self.model_path, f)):
                return False, f"Missing file: {f}"
        return True, "OK"
        
    def run(self):
        model_check, model_msg = self.check_model()
        self.debug_text.emit(f"Vosk Model: {model_msg}")
        
        if not model_check:
            self.status.emit("Vosk Model Error")
            return
            
        try:
            self.model = Model(self.model_path)
            self.rec = KaldiRecognizer(self.model, self.fs)
            self.debug_text.emit("Vosk: Loaded")
        except Exception as e:
            self.debug_text.emit(f"Vosk Error: {e}")
            self.status.emit(f"Error: {e}")
            return
        
        self.debug_text.emit(f"Settings: FS={self.fs}Hz, Gain={self.gain}x, Threshold={self.sensitivity}%")
        
        try:
            devices = sd.query_devices()
            self.debug_text.emit(f"Devices: {len(devices)}")
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    marker = " *" if i == sd.default.device[0] else ""
                    self.debug_text.emit(f"  [{i}]{marker} {d['name']}")
        except Exception as e:
            self.debug_text.emit(f"Device list error: {e}")
        
        def audio_callback(indata, frames, time_info, status):
            self.callback_count += 1
            
            if status and 'overflow' in str(status).lower():
                if self.callback_count % 200 == 0:
                    self.debug_text.emit(f"Audio buffer overflow")
                return
            
            try:
                rms = np.sqrt(np.mean(indata.astype(np.float32)**2)) / 32768.0
                level_display = min(1.0, rms * self.gain * 10)
                
                if self.callback_count % 5 == 0:
                    self.audio_level.emit(float(level_display))
                
                audio_int16 = indata.astype(np.int16)
                if self.rec.AcceptWaveform(audio_int16.tobytes()):
                    result = self.rec.Result()
                    text = json.loads(result).get("text", "").strip()
                    
                    if text:
                        self.debug_text.emit(f"Recognized: '{text}'")
                        self._process_text(text)
                else:
                    if self.callback_count % 100 == 0:
                        partial = self.rec.PartialResult()
                        partial_text = json.loads(partial).get("partial", "").strip()
                        if partial_text and len(partial_text) > 3:
                            self.debug_text.emit(f"...{partial_text}")
            except Exception as e:
                if self.callback_count % 50 == 0:
                    self.debug_text.emit(f"Audio callback error: {e}")
                        
        try:
            self.stream = sd.InputStream(
                channels=1, 
                samplerate=self.fs, 
                dtype='int16', 
                callback=audio_callback, 
                blocksize=int(self.fs * self.chunk_duration),
                latency='low'
            )
            self.stream.start()
            self.debug_text.emit(f"Stream Started: {self.fs}Hz")
            self.running = True
            
            while self.running and (not self.names or self.names.get_to_work()):
                QThread.msleep(50)
                
        except Exception as e:
            self.debug_text.emit(f"Stream Error: {e}")
            self.status.emit(f"Audio Error: {e}")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
            
    def _process_text(self, text):
        text_lower = text.lower().strip()
        
        if "найди" not in text_lower:
            self.debug_text.emit(f"No trigger 'найди' in text")
            return
        
        if self.names and self.names.add(text):
            self.command_detected.emit(text)
            self.status.emit(f"Command: {text}")
            self.debug_text.emit(f"Command Executed: '{text}'")
            self.debug_text.emit(f"Detection ACTIVATED!")
        else:
            self.status.emit(f"Not recognized: {text}")
            self.debug_text.emit(f"Object not in dictionary: '{text}'")
            
        self.recognized.emit(text)
            
    def stop(self):
        self.running = False
        if self.stream and self.stream.active:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        self.wait(1000)


class VideoAnalyzer(QThread):
    frame_ready = pyqtSignal(QPixmap, float, float, str)
    fps_updated = pyqtSignal(float)
    detection_status = pyqtSignal(str)
    model_status = pyqtSignal(str)
    loading_progress = pyqtSignal(int)
    debug_text = pyqtSignal(str)
    detection_saved = pyqtSignal(dict)
    
    def __init__(self, model_name=YOLO_MODEL_NAME, width=640, height=480, names_manager=None):
        super().__init__()
        self.model_name = model_name
        self.width = width
        self.height = height
        self.names = names_manager
        self.cap = None
        self.model = None
        self.running = False
        self.fps = 0
        self.last_time = None
        self.zoom_level = 1.0
        self.zoom_center = None
        self.detecting = False
        self.target_classes = []
        self.target_names = []
        self.command_text = ""
        self.frame_count = 0
        self.detections = []
        self.last_detection_time = 0
        
    def init_model(self):
        from ultralytics import YOLO
        
        self.model_status.emit(f"Loading YOLO: {self.model_name}...")
        self.loading_progress.emit(30)
        
        try:
            self.model = YOLO(self.model_name)
            self.loading_progress.emit(70)
            
            if torch.cuda.is_available():
                self.model.to('cuda')
                self.model_status.emit("YOLO Loaded (CUDA)")
            else:
                self.model_status.emit("YOLO Loaded (CPU)")
                
            self.loading_progress.emit(100)
            return True
            
        except Exception as e:
            self.model_status.emit(f"YOLO Error: {e}")
            self.debug_text.emit(f"YOLO load error: {e}")
            return False
        
    def run(self):
        camera_opened = False
        for cam_id in [0, 1, 2]:
            self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
            if self.cap.isOpened():
                self.debug_text.emit(f"Camera opened: index {cam_id}")
                camera_opened = True
                break
            self.cap.release()
        
        if not camera_opened:
            self.model_status.emit("Failed to open camera")
            self.debug_text.emit("Camera init failed")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.init_model():
            self.cap.release()
            return
        
        self.running = True
        self.last_time = cv2.getTickCount()
        self.frame_count = 0
        
        self.debug_text.emit("=== VIDEO LOOP STARTED ===")
        
        while self.running and self.cap.isOpened() and (not self.names or self.names.get_to_work()):
            ret, frame = self.cap.read()
            if not ret:
                self.debug_text.emit("Failed to read frame")
                break
            
            self.frame_count += 1
            display_status = ""
            current_time = time.time()
            
            if self.names:
                updated, classes = self.names.get_status()
                
                if updated:
                    self.target_classes = classes
                    self.target_names = self.names.get_en_names()
                    self.detecting = len(classes) > 0
                    
                    self.debug_text.emit(f"=== DETECTION STATUS UPDATE ===")
                    self.debug_text.emit(f"  detecting={self.detecting}")
                    self.debug_text.emit(f"  target_names={self.target_names}")
                    self.debug_text.emit(f"  model={self.model is not None}")
                    
                    if self.detecting:
                        self.command_text = self.names.get_last_command()
                        names_str = ', '.join(self.target_names) if self.target_names else str(classes)
                        status_msg = f"Searching: {names_str}"
                        self.detection_status.emit(status_msg)
                        self.debug_text.emit(f"Detection ACTIVATED: {names_str}")
                        
                        if self.model and self.target_names:
                            try:
                                self.model.set_classes(self.target_names)
                                self.debug_text.emit(f"YOLO-World classes SET: {self.target_names}")
                            except Exception as e:
                                self.debug_text.emit(f"set_classes error: {e}")
                    else:
                        self.detection_status.emit("Waiting for command")
            
            if self.detecting and self.target_names and self.model:
                try:
                    with torch.inference_mode():
                        results = self.model(
                            frame, 
                            verbose=False,
                            conf=0.15,
                            iou=0.7
                        )
                    
                    if results and len(results) > 0 and len(results[0]) > 0:
                        frame = self._draw_detections(frame, results[0])
                        display_status = f"DETECTED: {len(results[0])}"
                        
                        if current_time - self.last_detection_time > 1.0:
                            self._save_detections_simple(frame, results[0])
                            self.last_detection_time = current_time
                            
                except Exception as e:
                    if self.frame_count % 30 == 0:
                        self.debug_text.emit(f"Detection ERROR: {e}")
            else:
                if self.frame_count % 30 == 0:
                    self.debug_text.emit(f"[Frame {self.frame_count}] Detection SKIPPED")
                cv2.putText(frame, "SAY: FIND <OBJECT>", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            curr = cv2.getTickCount()
            if self.last_time and curr != self.last_time:
                self.fps = 0.9 * self.fps + 0.1 * (cv2.getTickFrequency() / (curr - self.last_time))
            self.last_time = curr
            if self.frame_count % 15 == 0:
                self.fps_updated.emit(self.fps)
            
            self._send_frame(frame, display_status)
        
        if self.cap:
            self.cap.release()
        self.debug_text.emit("Video stream stopped")
        
    def _draw_detections(self, frame, results):
        for i in range(len(results)):
            try:
                conf = results.boxes.conf[i].item()
                if conf < 0.15:
                    continue
                    
                x1, y1, x2, y2 = results.boxes.xyxy[i].to(torch.int).tolist()
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                cls = int(results.boxes.cls[i].item())
                name = self.target_names[cls] if cls < len(self.target_names) else f"class_{cls}"
                label = f"{name} {conf:.2f}"
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-25), (x1+tw, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
            except Exception as e:
                self.debug_text.emit(f"Error drawing box {i}: {e}")
        
        return frame
        
    def _save_detections_simple(self, frame, results):
        detections = []
        for i in range(len(results)):
            conf = results.boxes.conf[i].item()
            if conf < 0.15:
                continue
            x1, y1, x2, y2 = results.boxes.xyxy[i].to(torch.int).tolist()
            cls = int(results.boxes.cls[i].item())
            name = self.target_names[cls] if cls < len(self.target_names) else f"class_{cls}"
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            detections.append({
                'name': name,
                'confidence': round(conf, 3),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': [cx, cy],
                'timestamp': datetime.now().isoformat()
            })
        
        for det in detections:
            self.detections.append(det)
            self.detection_saved.emit(det)
        
        try:
            with open(DETECTIONS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    'total_detections': len(self.detections),
                    'last_updated': datetime.now().isoformat(),
                    'detections': self.detections[-100:]
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.debug_text.emit(f"Save error: {e}")
        
    def _send_frame(self, frame, status_text=""):
        if self.zoom_level > 1.0 and self.zoom_center:
            frame = self._apply_zoom(frame)
        
        if status_text:
            cv2.putText(frame, status_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        
        pixmap = QPixmap.fromImage(qimg)
        self.frame_ready.emit(pixmap, float(w), float(h), status_text)
        
    def _apply_zoom(self, frame):
        h, w = frame.shape[:2]
        cx, cy = int(self.zoom_center[0]), int(self.zoom_center[1])
        new_w, new_h = int(w / self.zoom_level), int(h / self.zoom_level)
        
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)
        
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
    def set_zoom(self, level, center_x, center_y):
        self.zoom_level = max(1.0, min(3.0, level))
        self.zoom_center = (center_x, center_y)
        
    def reset_zoom(self):
        self.zoom_level = 1.0
        self.zoom_center = None
        
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait(2000)


class ClickableLabel(QLabel):
    left_clicked = pyqtSignal(object)
    right_clicked = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.current_pixmap = None
        self.frame_width = 640
        self.frame_height = 480
        
    def set_frame_size(self, w, h):
        self.frame_width = w
        self.frame_height = h
        
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.left_clicked.emit(event)
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Object Detector")
        self.setMinimumSize(1000, 850)
        
        self.names = NamesManager()
        self.names.load_commands()
        
        self.video_thread = None
        self.audio_thread = None
        
        self.last_click = None
        self.detection_count = 0
        
        self._init_ui()
        self._start_threads()
        
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        self.loading_bar = QProgressBar()
        self.loading_bar.setFixedHeight(8)
        self.loading_bar.setStyleSheet("""
            QProgressBar { background: #1a1a2e; border: none; border-radius: 4px; }
            QProgressBar::chunk { background: #89b4fa; border-radius: 4px; }
        """)
        self.loading_bar.setVisible(False)
        layout.addWidget(self.loading_bar)
        
        self.video_label = ClickableLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel { 
                background: #1a1a2e; 
                border: 2px solid #4a4a6a; 
                border-radius: 8px; 
                font-family: 'Segoe UI', Arial;
                color: #89b4fa;
            }
        """)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Loading model...")
        self.video_label.left_clicked.connect(self._on_left_click)
        self.video_label.right_clicked.connect(self._on_right_click)
        layout.addWidget(self.video_label)
        
        audio_group = QGroupBox("Microphone")
        audio_layout = QHBoxLayout()
        
        self.audio_meter = AudioLevelMeter()
        self.meter_label = QLabel("0%")
        self.meter_label.setFont(QFont("Segoe UI", 9))
        
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(1, 30)
        self.gain_slider.setValue(10)
        self.gain_slider.setFixedWidth(100)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        self.gain_label = QLabel("10.0x")
        self.gain_label.setStyleSheet("color: #4ecca3;")
        
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(1, 30)
        self.sens_slider.setValue(3)
        self.sens_slider.setFixedWidth(100)
        self.sens_slider.valueChanged.connect(self._on_sensitivity_changed)
        self.sens_label = QLabel("3%")
        self.sens_label.setStyleSheet("color: #4ecca3;")
        
        audio_layout.addWidget(QLabel("Gain:"))
        audio_layout.addWidget(self.gain_slider)
        audio_layout.addWidget(self.gain_label)
        audio_layout.addWidget(QLabel("Threshold:"))
        audio_layout.addWidget(self.sens_slider)
        audio_layout.addWidget(self.sens_label)
        audio_layout.addWidget(self.audio_meter, stretch=1)
        audio_layout.addWidget(self.meter_label)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        status_layout = QHBoxLayout()
        self.status_text = QLabel("Initializing...")
        self.status_text.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.status_text.setStyleSheet("color: #ffffff;")
        status_layout.addWidget(self.status_text)
        status_layout.addStretch()
        self.fps_text = QLabel("FPS: --")
        self.fps_text.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.fps_text.setStyleSheet("color: #4ecca3;")
        status_layout.addWidget(self.fps_text)
        layout.addLayout(status_layout)
        
        btn_layout = QHBoxLayout()
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedHeight(35)
        self.btn_clear.clicked.connect(self._on_clear)
        btn_layout.addWidget(self.btn_clear)
        
        self.btn_coords = QPushButton("Coords: --")
        self.btn_coords.setFixedHeight(35)
        self.btn_coords.setEnabled(False)
        btn_layout.addWidget(self.btn_coords)
        
        self.btn_zoom = QPushButton("Zoom: 1.0x")
        self.btn_zoom.setFixedHeight(35)
        self.btn_zoom.clicked.connect(self._reset_zoom)
        btn_layout.addWidget(self.btn_zoom)
        
        self.btn_save = QPushButton("Save Detections")
        self.btn_save.setFixedHeight(35)
        self.btn_save.clicked.connect(self._save_detections)
        btn_layout.addWidget(self.btn_save)
        
        self.btn_test = QPushButton("Test Mic")
        self.btn_test.setFixedHeight(35)
        self.btn_test.clicked.connect(self._test_mic)
        btn_layout.addWidget(self.btn_test)
        
        self.btn_debug_cmd = QPushButton("Test: 'bottle'")
        self.btn_debug_cmd.setFixedHeight(35)
        self.btn_debug_cmd.clicked.connect(self._debug_command)
        btn_layout.addWidget(self.btn_debug_cmd)
        
        btn_layout.addStretch()
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedHeight(35)
        self.btn_stop.setStyleSheet("background: #d20f39; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.btn_stop)
        
        layout.addLayout(btn_layout)
        
        # help_box = QGroupBox("Instructions")
        # help_layout = QVBoxLayout()
        # help_text = QLabel(
        #     "1. Say clearly: 'FIND BOTTLE' (or another object)\n"
        #     "2. Detection continues until you press 'Clear'\n"
        #     "3. Supported: bottle, laptop, cup, cell phone, keyboard, mouse, clock, book, person\n"
        #     "4. Right click = 2x zoom, Left click = show coordinates\n"
        #     "5. GREEN BOXES will appear around detected objects"
        # )
        # help_text.setWordWrap(True)
        # help_text.setStyleSheet("color: #a6adc8;")
        # help_layout.addWidget(help_text)
        # help_box.setLayout(help_layout)
        # layout.addWidget(help_box)
        
        self.log = QTextEdit()
        self.log.setFixedHeight(200)
        self.log.setFont(QFont("Consolas", 8))
        self.log.setReadOnly(True)
        self.log.setStyleSheet("""
            QTextEdit { 
                background: #0a0a15; 
                color: #89b4fa; 
                border: 1px solid #313244; 
                border-radius: 4px; 
            }
        """)
        layout.addWidget(self.log)
        
    def _start_threads(self):
        self.video_thread = VideoAnalyzer(
            model_name=YOLO_MODEL_NAME,
            width=640, height=480,
            names_manager=self.names
        )
        self.video_thread.frame_ready.connect(self._update_video)
        self.video_thread.fps_updated.connect(self._update_fps)
        self.video_thread.detection_status.connect(self._update_status)
        self.video_thread.model_status.connect(self._on_model_status)
        self.video_thread.loading_progress.connect(self._on_loading_progress)
        self.video_thread.debug_text.connect(self._log)
        self.video_thread.detection_saved.connect(self._on_detection_saved)
        self.video_thread.start()
        
        self.audio_thread = AudioRecorder(
            model_path=VOSK_MODEL_PATH,
            fs=16000,
            names_manager=self.names
        )
        self.audio_thread.audio_level.connect(self._update_level)
        self.audio_thread.command_detected.connect(self._on_command)
        self.audio_thread.debug_text.connect(self._log)
        self.audio_thread.start()
        
    def _on_model_status(self, msg):
        self.status_text.setText(msg)
        self._log(msg)
        
    def _on_loading_progress(self, value):
        self.loading_bar.setValue(value)
        self.loading_bar.setVisible(value < 100)
        if value >= 100:
            self._log("Model ready")
            self.status_text.setText("Say: 'FIND <object>'")
        
    def _update_level(self, level):
        self.audio_meter.set_level(level)
        self.meter_label.setText(f"{int(level*100)}%")
        
    def _on_gain_changed(self, value):
        if self.audio_thread:
            self.audio_thread.set_gain(value)
            self.gain_label.setText(f"{value/1.0:.1f}x")
        
    def _on_sensitivity_changed(self, value):
        if self.audio_thread:
            self.audio_thread.set_sensitivity(value)
            self.sens_label.setText(f"{value}%")
        
    def _update_video(self, pixmap, w, h, status_text):
        self.video_label.set_frame_size(int(w), int(h))
        self.video_label.current_pixmap = pixmap
        scaled = pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)
        
    def _update_fps(self, fps):
        self.fps_text.setText(f"FPS: {fps:.1f}")
        
    def _update_status(self, msg):
        self.status_text.setText(msg)
        
    def _on_command(self, text):
        self._log(f"Command: '{text}'")
        self._log(f"Detection activated!")
        
    def _on_detection_saved(self, det):
        self.detection_count += 1
        self._log(f"Saved: {det['name']} at ({det['center'][0]}, {det['center'][1]}) conf={det['confidence']}")
        
    def _debug_command(self):
        self._log("Manual command: 'find bottle'")
        if self.names.add("найди бутылка"):
            self._log("Command accepted, detection activated")
            self.status_text.setText("Searching: bottle")
        else:
            self._log("Command rejected")
        
    def _on_left_click(self, event):
        if not self.video_label.current_pixmap:
            return
        px, py = event.pos().x(), event.pos().y()
        pm = self.video_label.current_pixmap
        
        scale = min(
            self.video_label.width() / pm.width(), 
            self.video_label.height() / pm.height()
        )
        ox = (self.video_label.width() - pm.width() * scale) / 2
        oy = (self.video_label.height() - pm.height() * scale) / 2
        vx = int((px - ox) / scale)
        vy = int((py - oy) / scale)
        
        if 0 <= vx < self.video_label.frame_width and 0 <= vy < self.video_label.frame_height:
            self.last_click = (vx, vy)
            self.btn_coords.setText(f"({vx}, {vy})")
            self.btn_coords.setEnabled(True)
            self._log(f"Click: ({vx}, {vy})")
            
    def _on_right_click(self, event):
        if not self.last_click:
            self._log("First click left button to select zoom point")
            return
        self.video_thread.set_zoom(2.0, self.last_click[0], self.last_click[1])
        self.btn_zoom.setText("Zoom: 2.0x")
        self._log(f"Zoom on ({self.last_click[0]}, {self.last_click[1]})")
        
    def _reset_zoom(self):
        self.video_thread.reset_zoom()
        self.btn_zoom.setText("Zoom: 1.0x")
        self._log("Zoom reset")
        
    def _on_clear(self):
        self.names.clear()
        self.status_text.setText("Waiting for command...")
        self.btn_coords.setText("--")
        self.btn_coords.setEnabled(False)
        self._reset_zoom()
        self._log("Cleared")
        
    def _save_detections(self):
        try:
            if os.path.exists(DETECTIONS_FILE):
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Detections", "detections_export.json", "JSON Files (*.json)"
                )
                if save_path:
                    with open(DETECTIONS_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    self._log(f"Saved to: {save_path}")
            else:
                self._log("No detections to save")
        except Exception as e:
            self._log(f"Save error: {e}")
        
    def _on_stop(self):
        self._log("Stopping...")
        if self.names:
            self.names.set_to_work(False)
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
        self.close()
        
    def _test_mic(self):
        self._log("=== MIC TEST ===")
        self._log("Speak into microphone now...")
        QMessageBox.information(self, "Mic Test", 
            "Speak into microphone!\n\n"
            "Level bar should move\n"
            "Say clearly: FIND BOTTLE\n"
            "Increase 'Gain' if bar doesn't respond")
        
    def _log(self, msg):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log.append(f"[{timestamp}] {msg}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        
    def closeEvent(self, event):
        self._on_stop()
        event.accept()


if __name__ == "__main__":
    if os.name == 'nt':
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    app.setStyleSheet("""
        QMainWindow { background: #1e1e2e; }
        QWidget { background: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', Arial; }
        QPushButton { 
            background: #313244; 
            color: #cdd6f4; 
            border: 1px solid #45475a; 
            border-radius: 6px; 
            padding: 6px 14px;
            font-weight: 500;
        }
        QPushButton:hover { background: #45475a; }
        QPushButton:pressed { background: #585b70; }
        QLabel { font-size: 10pt; }
        QGroupBox { 
            border: 1px solid #313244; 
            border-radius: 6px; 
            margin-top: 8px; 
            font-weight: bold; 
            color: #89b4fa;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QSlider::groove:horizontal { background: #313244; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #89b4fa; width: 14px; border-radius: 7px; margin: -4px 0; }
        QTextEdit { selection-background-color: #45475a; }
        QProgressBar { text-align: center; }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
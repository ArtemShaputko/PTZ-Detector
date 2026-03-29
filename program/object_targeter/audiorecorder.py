import sounddevice as sd
import numpy as np
import vosk
import threading
from pynput import keyboard
import json
from scipy.io import wavfile
import time
from names import Names
from zoom import ZoomController
from commands import CommandParser, CommandType
from logger import Logger


class AudioRecorder:
    def __init__(self, names: Names, zoom: ZoomController,
                 logger: Logger | None = None,
                 model_name = "vosk-model-small-ru-0.22",
                 fs = 44100,
                 device: int | str | None = 7):

        self.__names = names
        self.__zoom = zoom
        self.__parser = CommandParser()
        self.__logger = logger

        self.__fs = fs
        self.__device = device

        self.__model = vosk.Model(model_name)
        self.__stop_event = threading.Event()
        self.__listen_thread = None

        if self.__logger:
            self.__logger.info(f"fs: {self.__fs}")
            info = sd.query_devices(device) if device is not None else sd.query_devices(sd.default.device[0])
            self.__logger.info(f"Микрофон: {info['name']}")

    def __handle_command(self, command, text):
        if self.__logger:
            self.__logger.info(f"Команда: {command.type.name} — '{text}'")
        if command.type == CommandType.ZOOM_IN:
            self.__zoom.zoom_in()
        elif command.type == CommandType.ZOOM_OUT:
            self.__zoom.zoom_out()
        elif command.type == CommandType.EXIT:
            self.__names.set_to_work(False)
        elif command.type == CommandType.ADD:
            self.__names.add(command.text)
        elif command.type == CommandType.PLACE:
            self.__names.place(command.text)

    def __listen_loop(self):
        rec = vosk.KaldiRecognizer(self.__model, self.__fs)
        handled = False  # флаг чтобы не срабатывало дважды на одну фразу

        TRIGGERS = ["найти", "найди", "поиск",
                    "добавить", "добавь",
                    "приблизить", "приблизь", "увеличить",
                    "отдалить", "отдали", "уменьшить",
                    "выйти", "выход", "стоп"]

        handled = False
        handled_time = 0.0  # ← добавить

        def callback(indata, frames, t, status):
            nonlocal handled
            if self.__stop_event.is_set():
                return

            if rec.AcceptWaveform(indata.tobytes()):
                # фраза закончена — обрабатываем финальный текст
                text = json.loads(rec.Result()).get("text", "").strip().lower()
                handled = False

                if not text:
                    return

                if self.__logger:
                    self.__logger.info(f"Found text {text}")

                for trigger in TRIGGERS:
                    idx = text.find(trigger)
                    if idx != -1:
                        after = text[idx + len(trigger):].strip()
                        full = trigger + (" " + after if after else "")
                        command = self.__parser.parse(full)
                        if command.type != CommandType.UNKNOWN:
                            self.__handle_command(command, full)
                        break

        with sd.InputStream(callback=callback, channels=1,
                            samplerate=self.__fs, dtype='int16',
                            blocksize=int(self.__fs * 0.3),
                            device=self.__device):
            self.__stop_event.wait()

        if self.__logger:
            self.__logger.info("Exit listen loop")

    def stop(self):
        self.__stop_event.set()
        if self.__listen_thread and self.__listen_thread.is_alive():
            self.__listen_thread.join(timeout=3)
            if self.__listen_thread.is_alive() and self.__logger:
                self.__logger.warning("Listen поток не завершился вовремя.")

    def get_class(self):
        self.__stop_event.clear()
        self.__listen_thread = threading.Thread(
            target=self.__listen_loop, daemon=True
        )
        self.__listen_thread.start()

        print("Голосовые команды:\n"
              "\t'найти <предмет>'     - заменить искомые объекты\n"
              "\t'добавить <предмет>'  - добавить объект\n"
              "\t'приблизить'          - приблизить\n"
              "\t'отдалить'            - отдалить\n"
              "\t'стоп'                - выйти\n"
              "Клавиши: 'з' - заменить, 'д' - добавить, 'в' - выйти.")

        while self.__names.get_to_work():
            time.sleep(0.1)

        self.__stop_event.set()
        self.__listen_thread.join(timeout=3)
        if self.__listen_thread.is_alive() and self.__logger:
            self.__logger.warning("Listen поток не завершился вовремя.")

        if self.__logger:
            self.__logger.info("Exit Audio Recorder")
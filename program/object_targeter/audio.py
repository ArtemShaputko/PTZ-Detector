import sounddevice as sd
import vosk
import threading
import json
import time
from ru_word2number import w2n

from interfaces import IWorkConfigManager, IZoomController, CommandType, ICommandParser, ILogger, IAudioRecorder

from threadmanager import ThreadManager

class AudioRecorder(ThreadManager, IAudioRecorder):
    def __init__(self, config_manager: IWorkConfigManager, zoom: IZoomController, parser: ICommandParser,
                 logger: ILogger | None = None,
                 model_name = "vosk-model-small-ru-0.22",
                 fs = 44100,
                 device: int | str | None = 7):
        super().__init__()

        self.__config_manager = config_manager
        self.__zoom = zoom
        self.__parser = parser
        self.__logger = logger

        self.__fs = fs
        self.__device = device

        self.__model = vosk.Model(model_name)
        self.__listen_thread = None

        if self.__logger:
            self.__logger.info(f"fs: {self.__fs}")
            info = sd.query_devices(device) if device is not None else sd.query_devices(sd.default.device[0])
            self.__logger.info(f"Микрофон: {info['name']}")
            
    def __words_to_num(self, text: str) -> int | float | None:
        try:
            return w2n.word_to_num(text)
        except Exception as e:
            if self.__logger:
                self.__logger.warning(f"Convert words to num exception {e}")
        return None

    def __handle_command(self, command, text):
        if self.__logger:
            self.__logger.info(f"Команда: {command.type.name} — '{text}'")
        if command.type == CommandType.ZOOM_IN:
            self.__zoom.zoom_in()
        elif command.type == CommandType.ZOOM_OUT:
            self.__zoom.zoom_out()
        elif command.type == CommandType.EXIT:
            self.__config_manager.stop()
        elif command.type == CommandType.ADD:
            self.__config_manager.add(command.text)
        elif command.type == CommandType.REMOVE:
            self.__config_manager.remove_last()
        elif command.type == CommandType.PLACE:
            self.__config_manager.place(command.text)
        elif command.type == CommandType.FOLLOW:
            self.__config_manager.target_track = self.__words_to_num(command.text)
        elif command.type == CommandType.CONF:
            conf = self.__words_to_num(command.text)
            if conf:
                self.__config_manager.conf = conf / 100
            

    def __listen_loop(self):
        rec = vosk.KaldiRecognizer(self.__model, self.__fs)
        handled = False

        TRIGGERS = self.__parser.get_triggers()

        handled = False

        def callback(indata, frames, t, status):
            nonlocal handled
            if self._stop_event.is_set():
                return

            if rec.AcceptWaveform(indata.tobytes()):
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
            self._stop_event.wait()

        if self.__logger:
            self.__logger.info("Exit listen loop")

    def stop(self):
        self._stop_event.set()
        if self.__listen_thread and self.__listen_thread.is_alive():
            self.__listen_thread.join(timeout=3)
            if self.__listen_thread.is_alive() and self.__logger:
                self.__logger.warning("Listen поток не завершился вовремя.")

    def start(self):
        self._stop_event.clear()
        self.__listen_thread = threading.Thread(
            target=self.__listen_loop, daemon=False
        )
        self.__listen_thread.start()

        print("Голосовые команды:\n"
              "\t'найти <предмет>'       - заменить искомые объекты\n"
              "\t'добавить <предмет>'    - добавить объект\n"
              "\t'следить <id>'          - фокусировать вниание на находку id\n"
              "\t'уверенность <процент>' - задать порог уверенности для модели\n"
              "\t'приблизить'            - приблизить\n"
              "\t'отдалить'              - отдалить\n"
              "\t'стоп'                  - выйти\n")

        while self.__config_manager.to_work:
            time.sleep(0.1)

        self._stop_event.set()
        self.__listen_thread.join(timeout=3)
        if self.__listen_thread.is_alive() and self.__logger:
            self.__logger.warning("Listen поток не завершился вовремя.")

        if self.__logger:
            self.__logger.info("Exit Audio Recorder")
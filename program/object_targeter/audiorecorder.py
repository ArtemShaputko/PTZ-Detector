import sounddevice as sd
import numpy as np
import vosk
from pynput import keyboard
import json
from scipy.io import wavfile
import time
from names import Names
from zoom import ZoomController
from commands import CommandParser, CommandType
from program.object_targeter.logger import Logger

class AudioRecorder:
    def __init__(self, names: Names, zoom: ZoomController, logger: Logger | None = None, model_name = "vosk-model-small-ru-0.22", fs = 44100):
        
        self.__names = names
        self.__zoom = zoom
        self.__parser = CommandParser()
        self.__logger = logger
        
        self.fs = fs
        self.__model = vosk.Model(model_name)
        self.__rec = vosk.KaldiRecognizer(self.__model, fs)
        self.__audio_data = []
        self.__recording = False
        self.__to_add = False

    def __recognize_audio(self, audio_data):
        start_time = time.time()
        self.__rec.AcceptWaveform(audio_data.tobytes())
        result = self.__rec.Result()
        text = json.loads(result).get("text", "")
        if self.__logger:
            self.__logger.info(f"Speech recognition time: {time.time()-start_time}")
        return text

    def __callback(self, indata, frames, time, status):
        if self.__recording:
            self.__audio_data.append(indata.copy())

    def __start_recording(self):
        self.__recording = True
        self.__audio_data = []
        if self.__logger:
            self.__logger.info("Запись начата. Нажмите 'с' для остановки.")

    def __stop_recording(self):
        if self.__recording:
            self.__recording = False
            if self.__logger:
                self.__logger.info("Запись остановлена.")
            audio_array = np.concatenate(self.__audio_data).flatten()
            wavfile.write('output.wav', self.fs, audio_array)
            recognized_text = self.__recognize_audio(audio_array)
            command = self.__parser.parse(recognized_text, to_add=self.__to_add)

            if self.__logger:
                self.__logger.info(f"Команда: {command.type.name} — '{recognized_text}'")

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
    
    def __on_press(self, key):
        try:
            if key == keyboard.KeyCode.from_char('з') or key == keyboard.KeyCode.from_char('д'):
                self.__to_add = key == keyboard.KeyCode.from_char('д')
                self.__start_recording()
                if self.__logger:
                    self.__logger.info("Запись активирована.")
            elif key == keyboard.KeyCode.from_char('с'):
                self.__stop_recording()
                return False
            elif key == keyboard.KeyCode.from_char('в'):
                self.__names.set_to_work(False)
                return False
        except AttributeError:
            pass

    def get_class(self):
        while(self.__names.get_to_work()):
            with sd.InputStream(callback=self.__callback, channels=1, samplerate=self.fs, dtype='int16') as inputStream:
                print("Нажмите:\n\t'з' - заменить искомые объекты\n\t'д' - добавить объект\n\t'в' - выйти.")
                listener = keyboard.Listener(on_press=self.__on_press)
                listener.start()
                listener.join()
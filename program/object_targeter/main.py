import threading
from vosk import SetLogLevel
from names import Names
from serialwriter import SerialWriter
from zoom import ZoomController
from preprocessor import Preprocessor
from audiorecorder import AudioRecorder
from video_analyze import VideoAnalyzer
from logger import Logger

class Platform:
    def __init__(self, size: tuple[int, int]):
        SetLogLevel(-1)
        self.__size = size
        self.__logger = Logger()
        self.__names = Names(logger = self.__logger)
        self.__writer = SerialWriter(logger = self.__logger, size=self.__size)
        self.__zoom = ZoomController(writer= self.__writer, logger = self.__logger, min_zoom=1.0, max_zoom=5.0, step=0.5, size = self.__size)
        self.__preprocessor = Preprocessor(use_clahe=False, use_bilateral=False)
        self.__recorder = AudioRecorder(names=self.__names, zoom=self.__zoom, logger=self.__logger)
        self.__analyzer = VideoAnalyzer(names=self.__names, zoom=self.__zoom, logger= self.__logger,
                                        serial_writer=self.__writer,
                                        preprocessor=self.__preprocessor,
                                        size=self.__size)
        self.__threads = [
            threading.Thread(target=self.__analyzer.start),
            threading.Thread(target=self.__recorder.get_class),
            threading.Thread(target=self.__writer.write_loop),
        ]

    def run(self):
        for t in self.__threads:
            t.start()
            
        try:
            for t in self.__threads:
                t.join()
        except Exception as e:
            if self.__logger:
                self.__logger.info(f"\nОстановка: {e}")
        finally:
            self.__names.set_to_work(False)
            self.__writer.stop()
            for t in self.__threads:
                t.join(timeout=3)
            if self.__logger:
                self.__logger.info("Завершено.")

if __name__ == '__main__':
    Platform((1280, 720)).run()
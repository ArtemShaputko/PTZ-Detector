import threading
from vosk import SetLogLevel
from names import Names
from serialwriter import SerialWriter
from zoom import ZoomController
from preprocessor import Preprocessor
from audiorecorder import AudioRecorder
from video_analyze import VideoAnalyzer
from logs import Logger

class Platform:
    def __init__(self):
        SetLogLevel(-1)
        self.__logger = Logger()
        self.__names = Names(logger = self.__logger)
        self.__writer = SerialWriter(logger = self.__logger)
        self.__zoom = ZoomController(logger = self.__logger, min_zoom=1.0, max_zoom=5.0, step=0.5)
        self.__preprocessor = Preprocessor(use_clahe=False, use_bilateral=False)
        self.__recorder = AudioRecorder(names=self.__names, zoom=self.__zoom, logger=self.__logger)
        self.__analyzer = VideoAnalyzer(names=self.__names, zoom=self.__zoom, logger= self.__logger,
                                        serial_writer=self.__writer,
                                        preprocessor=self.__preprocessor)
        self.__threads = [
            threading.Thread(target=self.__analyzer.start, daemon=True),
            threading.Thread(target=self.__recorder.get_class, daemon=True),
            threading.Thread(target=self.__writer.write_loop, daemon=True),
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
    Platform().run()
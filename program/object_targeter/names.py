import threading
from ultralytics.utils.plotting import Colors
import argostranslate.translate as translate
from model_names import en_model_names
import time
from logger import Logger


class Names:
    CONSUMER_IO = "io"
    CONSUMER_ANALYZER = "analyzer"

    def __init__(self, logger: Logger | None = None):
        self.translater = translate.get_translation_from_codes(from_code="ru", to_code="en")
        self.colors = Colors()
        self.current = {"": ""}
        self.to_work = True
        self.lock = threading.Lock()
        self.work_lock = threading.Lock()
        self.__logger = logger

        # Per-consumer updated flags
        self.__updated: dict[str, bool] = {
            self.CONSUMER_IO: True,
            self.CONSUMER_ANALYZER: True,
        }

    def place(self, ru_text):
        if ru_text is not None:
            translates = self.translater.hypotheses(ru_text)
            common = [x.value for x in translates if x.value in en_model_names]
            en_text = common[0] if common else translates[0].value
            with self.lock:
                self.current = {ru_text: en_text}
                for k in self.__updated:
                    self.__updated[k] = True

    def add(self, ru_text):
        if ru_text is not None:
            start_time = time.time()
            translates = self.translater.hypotheses(ru_text)
            common = [x.value for x in translates if x.value in en_model_names]
            en_text = common[0] if common else translates[0].value
            if self.__logger:
                self.__logger.info(f"translated: {en_text}, time = {time.time() - start_time}")
            with self.lock:
                self.current[ru_text] = en_text
                for k in self.__updated:
                    self.__updated[k] = True

    def get_names(self, consumer_id: str):
        with self.lock:
            updated = self.__updated.get(consumer_id, True)
            self.__updated[consumer_id] = False
            return updated, self.current.copy()

    def get_to_work(self):
        with self.work_lock:
            return self.to_work

    def set_to_work(self, to_work):
        with self.work_lock:
            self.to_work = to_work
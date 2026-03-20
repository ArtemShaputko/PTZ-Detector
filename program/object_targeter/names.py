import threading
from ultralytics.utils.plotting import Colors
import argostranslate.translate as translate
from model_names import en_model_names
import time
from logs import Logger


class Names:
    def __init__(self, logger: Logger | None = None):
        self.translater = translate.get_translation_from_codes(from_code="ru", to_code="en")
        self.colors = Colors()
        self.updated = True
        self.current = {"бутылка": "bottle", "телефон": "phone", "наушники": "headphones", "карандаш": "pencil"}
        self.to_work = True
        self.lock = threading.Lock()
        self.work_lock = threading.Lock()
        
        self.__logger = logger
                
    def place(self, ru_text):
        if ru_text is not None:
            translates = self.translater.hypotheses(ru_text)
            common = [x.value for x in translates if x.value in en_model_names]
            en_text = common[0] if common else translates[0].value
            with self.lock:
                self.current = {}
                self.current[ru_text] = en_text
                self.updated = True
                
    def add(self, ru_text):
        if ru_text is not None:
            start_time = time.time()
            translates = self.translater.hypotheses(ru_text)
            common = [x.value for x in translates if x.value in en_model_names]
            en_text = common[0] if common else translates[0].value
            if self.__logger:
                self.__logger.info(f"translated: {en_text}, time = {time.time()-start_time}")
            with self.lock:
                self.current[ru_text] = en_text
                self.updated = True
            
    def get_names(self):
        with self.lock:
            last_updated = self.updated
            self.updated = False
            return last_updated, self.current.copy()
    
    def get_to_work(self):
        with self.work_lock:
            return self.to_work
        
    def set_to_work(self, to_work):
        with self.work_lock:
            self.to_work = to_work

import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir,
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(filename),
                logging.StreamHandler()  # дублируем в консоль
            ]
        )
        self.__log = logging.getLogger("App")

    def info(self, msg: str):
        self.__log.info(msg)

    def warning(self, msg: str):
        self.__log.warning(msg)

    def error(self, msg: str):
        self.__log.error(msg)
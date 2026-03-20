from dataclasses import dataclass
from enum import Enum, auto

class CommandType(Enum):
    ZOOM_IN   = auto()
    ZOOM_OUT  = auto()
    PLACE     = auto()  # заменить объект
    ADD       = auto()  # добавить объект
    EXIT      = auto()
    UNKNOWN   = auto()

@dataclass
class Command:
    type: CommandType
    text: str = ""  # распознанный текст (для PLACE/ADD)

class CommandParser:
    __ZOOM_IN  = ["приблизить", "приблизь", "увеличить"]
    __ZOOM_OUT = ["удалить", "удали", "уменьшить", "отдалить"]
    __EXIT     = ["выйти", "выход", "стоп"]

    def parse(self, text: str, to_add: bool = False) -> Command:
        text = text.strip().lower()

        if any(w in text for w in self.__ZOOM_IN):
            return Command(CommandType.ZOOM_IN)
        if any(w in text for w in self.__ZOOM_OUT):
            return Command(CommandType.ZOOM_OUT)
        if any(w in text for w in self.__EXIT):
            return Command(CommandType.EXIT)
        if text:
            return Command(CommandType.ADD if to_add else CommandType.PLACE,
                           text=text)
        return Command(CommandType.UNKNOWN)
from dataclasses import dataclass
from enum import Enum, auto


class CommandType(Enum):
    ZOOM_IN  = auto()
    ZOOM_OUT = auto()
    PLACE    = auto()  # заменить объект
    ADD      = auto()  # добавить объект
    EXIT     = auto()
    UNKNOWN  = auto()


@dataclass
class Command:
    type: CommandType
    text: str = ""  # распознанный текст (для PLACE/ADD)


class CommandParser:
    __ZOOM_IN  = ["приблизить", "приблизь", "увеличить"]
    __ZOOM_OUT = ["отдалить", "отдали", "уменьшить", "удалить", "удали"]
    __EXIT     = ["выйти", "выход", "стоп"]
    __PLACE    = ["найти", "найди", "поиск"]
    __ADD      = ["добавить", "добавь"]

    def parse(self, text: str, to_add: bool = False) -> Command:
        text = text.strip().lower()

        if any(w in text for w in self.__ZOOM_IN):
            return Command(CommandType.ZOOM_IN)
        if any(w in text for w in self.__ZOOM_OUT):
            return Command(CommandType.ZOOM_OUT)
        if any(w in text for w in self.__EXIT):
            return Command(CommandType.EXIT)

        # голосовые префиксы — определяют тип и обрезают ключевое слово
        for w in self.__PLACE:
            if text.startswith(w):
                return Command(CommandType.PLACE, text=text[len(w):].strip())

        for w in self.__ADD:
            if text.startswith(w):
                return Command(CommandType.ADD, text=text[len(w):].strip())

        if text:
            return Command(CommandType.ADD if to_add else CommandType.PLACE, text=text)

        return Command(CommandType.UNKNOWN)
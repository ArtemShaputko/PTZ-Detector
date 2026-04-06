from abc import ABC, abstractmethod

class ThreadManager(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass
from abc import ABC, abstractmethod

class AbcSolver(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def set_const(self):
        pass
    
    @abstractmethod
    def transform(self):
        pass
    
    @abstractmethod
    def parse(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def info(self):
        pass
    
    @abstractmethod
    def cv(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
    
    
    @abstractmethod
    def solve(self):
        pass
    
'''Model does not do model stuff yet'''

import NNUtils
from NNUtils import PreprocessedData as PpD 
from NetLogPreprocessor import NetProcState as NPpS
from HostLogPreprocessor import HostProcState as HPpS


class CNN:
    def __init__(self): pass
    def Train(self, state: NPpS | HPpS, data: PpD): pass
    def Predict(self, state: NPpS | HPpS, data: PpD) -> float: pass

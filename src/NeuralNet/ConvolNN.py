'''Model does not do model stuff yet'''

import NNUtils
import numpy as np

from NNUtils import PreprocessedData as PpD, PreprocessingState as PpS, NNConfig as NNC 
from NetLogPreprocessor import NetProcState as NPpS
from HostLogPreprocessor import HostProcState as HPpS


class CNN:
    def __init__(self, config:NNC):
        self.config:NNC = config  
        
    def Train(self, state: NPpS | HPpS, data: PpD): pass
    def Predict(self, state: NPpS | HPpS, data: PpD) -> float: pass

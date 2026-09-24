'''The preprocessor exists as a completely seperate entity. Please do not make it a 
component of the NN or it'll spaghetify the codebase in a matter of minutes... '''

#? GOOD LUCK!!

import NNUtils
from NNUtils import PreprocessedData as PpD
from NNUtils import PreprocessingState as PpS

class ClusterNN:
    def __init__(self): pass
    def Train(self, state: PpS, data: PpD): pass
    def Predict(self, state: PpS, data: PpD) -> float: pass
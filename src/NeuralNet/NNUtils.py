import numpy as np

from dataclasses import dataclass
from pathlib import Path
from enum import Enum

class Model(Enum):
    UNK = 0
    CONVOLVUTIONAL = 1
    CLUSTERING = 2

#TODO: Add your log formats here
class Format(Enum):
    UNK = 0
    ADFALD = 1

#? Gets passed the preprocessor, any information the preproc requires to understand
#? what it's doing goes in here. Note that note all the variables need to be set. if you
#? reference the ThomTest.py the only values I care to initialize before passing it 
#? to the preprocessor is the model, format, a few booleans and the log name. The rest
#? can and should be derived from your preprocessor. After which this preproc state struct
#? can be used as the NN state information.
@dataclass
class PreprocessingState:
    model:Model         = Model.UNK
    logFormat:Format    = Format.UNK

    isHostLog:bool      = True
    isAnomolous:bool    = False
    isTraining:bool     = True
    doDebug:bool        = True
    
    logName:str | None  = None
    logPath:Path| None  = None

#? Throws a tantrum if you don't set something important
def ValidateConfig(config) -> bool:
    if not config.isHostLog:        raise ValueError ("Provided host preproc with netlog...")
    if config.logName is None:      raise ValueError ("No logName Provided...")
    if config.model is Model.UNK:   raise ValueError ("No Model Provided...")
    if config.logFormat is Format.UNK: raise ValueError ("No Format Provided...")

    return True


#? The class containing our data which we pass to the models
@dataclass
class PreprocessedData:
    x: object                                #* The data we're using
    y: float | None = None                   #* The value we're trying to predict 
                                             #* (0 = norm, 1 = abnorm)
    metadata: dict[str, any] | None   = None #* Defined by the preprocessor employed      


def ReLu(x:float) -> float: return max(0, x)
def Sigmoid(x:float) -> float: return 1/(1 + np.exp(-x))

#? A class representation of a neuron.
class Neuron:
    def __init__(self, weightVecLen:int):
        self.weights = [0.0] * weightVecLen #*E.g len = 4 = [0.0, 0.0, 0.0, 0.0]
        self.bias = 0.0

    def Output(self, inputs) -> float:
        total = self.bias
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i] #*Sum(i_0j_0, i_1j_1, etc etc) + bias

        return ReLu(total)
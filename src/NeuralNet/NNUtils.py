import random
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from enum import Enum

@dataclass
class Vec4:
    def __init__(self, randomize:bool = False, seed:int | None=None, 
                 w:float=0.0, x:float=0.0, y:float=0.0, z:float=0.0):
        if randomize:
            self.Randomize(seed)
            return
        
        self.w: float = 0.00
        self.x: float = 0.00
        self.y: float = 0.00
        self.z: float = 0.00

    #? If you want a predictable starting point enter a seed
    def Randomize(self, seed:int | None = None) -> None:
        if seed: random.seed(seed)

        self.w = round(random.uniform(-1.00, 1.00), 2)
        self.x = round(random.uniform(-1.00, 1.00), 2)
        self.y = round(random.uniform(-1.00, 1.00), 2)
        self.z = round(random.uniform(-1.00, 1.00), 2)

    def AsList(self) -> list[float]: return [self.w, self.x, self.y, self.z]
    def ToConsole(self) -> None: print(f"{self.w}, {self.x}, {self.y}, {self.z}")

    def Add(self, other:Vec4) -> None:
        self.w += other.w
        self.x += other.x
        self.y += other.y
        self.z += other.z


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
    model:Model                     = Model.UNK
    logFormat:Format                = Format.UNK
    vocabulary:dict[int, str] | None= None
    embedding:dict[int, Vec4] | None= None

    isHostLog:bool      = True
    isAnomolous:bool    = False
    isTraining:bool     = True
    doDebug:bool        = True
    
    logName:str | None  = None
    logPath:Path| None  = None

    def Debug(self):
        print("--------------  PREPROC STATE DEBUG --------------")
        print(f"\t- Model: {self.model.name}\n\t- Format: {self.logFormat.name}")
        print(f"\t- Training = {self.isTraining}")
        if(self.isTraining):print(f"\t- Log anomaly rating = {float(self.isAnomolous)}")
        if(self.isHostLog): print(f"\t- Host system log detected")
        else:               print(f"\t- Server log detected\n")

        print(f"\n-----------  Embedding & Vocabulary  ------------")
        for id, embed in self.embedding.items(): print(f"\t\t--> {id} : {self.vocabulary[id]} : {embed.AsList()}")

    

#? Throws a tantrum if you don't set something important
def ValidateConfig(config, printToConsole:bool = False) -> bool:
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

    def Debug(self) -> None:
        print("--------------- PREPROC DATA DEBUG  --------------")
        print(f"\t-X = {self.x}")
        print(f"\t-Y = {self.y}")

        if not self.metadata:
            print("\t-Metadata = None")
            return

        print(f"\t-Metadata")
        for id, value in self.metadata.items(): print(f"\t\t--> {id} : {value}")


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
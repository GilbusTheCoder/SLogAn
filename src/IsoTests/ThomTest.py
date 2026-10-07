''' 
Use this file for testing and importing whatever you need tested. We all have our 
own files to (hopefully) avoid a ton of merging that would arise when using a shared
main.py '''

import os
import sys

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                          ".."))
if PARENT_DIR not in sys.path: sys.path.append(PARENT_DIR)
SIBNET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                          "../NeuralNet"))
if SIBNET_DIR not in sys.path: sys.path.append(SIBNET_DIR)
ADFA_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                               "../../dat/HostLogs/ADFA-LD_Logs"))

import NeuralNet.HostLogPreprocessor as HLpP
import NeuralNet.NetLogPreprocessor as NLpP
from NeuralNet.ConvolNN import ConvolNN as CNN
from NeuralNet.NNUtils import PreprocessingState as PpS, PreprocessingData as PpD, LogFormat, Model, NNConfig, AnomalyType

from pathlib import Path
import random
attackDataDir = Path(f"{ADFA_DATA_DIR}/Attack_Data_Master")
trainingDataDir = Path(f"{ADFA_DATA_DIR}/Training_Data_Master")
validationDataDir = Path(f"{ADFA_DATA_DIR}/Validation_Data_Master")



cnnCfg = NNConfig(model=Model.CONVOLUTIONAL)
stateCfg = PpS (model=Model.CONVOLUTIONAL,
                logFormat=LogFormat.ADFALD, 
                isTraining=True, doDebug=False)

preprocessor = HLpP.HostLogPreprocessor(stateCfg)
trainingStateDat:tuple[PpS, PpD] = preprocessor.GetStateData()
model = CNN(config=cnnCfg, state=trainingStateDat[0], data=trainingStateDat[1])

def AddFilesToTrainingData(dir:Path, trainingData:list[Path]) -> list[Path]:
    newData = trainingData
    for f in dir.rglob("*.txt"):
        if f.is_file(): newData.append(f)

    return newData

def TrainModel(metaEpoch:int, epochs: int, files:list[Path], doShuffle:bool = True):
    data = files
    model.train()            #Tell the pytorch NN it's in training mode
    model.lossData.clear()
    
    for i in range(metaEpoch):        
        trainingData = data
        if doShuffle: random.shuffle(trainingData)
    
        while trainingData:
            file = trainingData.pop()        
            stateCfg.logName=f"{file.name}"
            stateCfg.logPath=file
            stateCfg.isAnomalous = stateCfg.DetIsAnomolous()
            if trainingDataDir.as_posix() in Path(file).as_posix():
                stateCfg.anomalyType = AnomalyType.NORMAL
            else: stateCfg.anomalyType = stateCfg.DetAnomalyType()

            preprocessor.Load(stateCfg)
            trainingStateDat = preprocessor.GetStateData()
            model.state = trainingStateDat[0]
            model.data = trainingStateDat[1]
            model.Train(epochs) 


def PredictModel(logDir:Path):
    for file in logDir.rglob("*.txt"):
        if file.is_file():        
            stateCfg.logName=f"{file.name}"
            stateCfg.logPath=file

            preprocessor.Load(stateCfg)
            trainingStateDat = preprocessor.GetStateData()
            model.state = trainingStateDat[0]
            model.data = trainingStateDat[1]

            print("\n---------- RESULTS ----------")
            print(f"\tLog: {stateCfg.logName}, {stateCfg.logPath}")
            print(f"\tResult: {model.Predict()}\n\n")


normalTrainingData:list[Path] = []
attackTrainingData:list[Path] = []
allTrainingData:list[Path] = []

AddFilesToTrainingData(trainingDataDir, normalTrainingData)
AddFilesToTrainingData(attackDataDir, attackTrainingData)
AddFilesToTrainingData(trainingDataDir, allTrainingData)
AddFilesToTrainingData(attackDataDir, allTrainingData)


TrainModel(1, 100, allTrainingData, True) 

model.state.isTraining = False
#PredictModel(trainingDataDir)
#PredictModel(attackDataDir)
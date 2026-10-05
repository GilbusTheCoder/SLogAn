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
from NeuralNet.NNUtils import PreprocessingState as PpS, PreprocessedData as PpD, Format, Model, NNConfig

from pathlib import Path
attackDataDir = Path(f"{ADFA_DATA_DIR}/Attack_Data_Master/Adduser_1")



cnnCfg = NNConfig(model=Model.CONVOLUTIONAL)
stateCfg = PpS(model=Model.CONVOLUTIONAL,
                logFormat=Format.ADFALD,
                logPath=None,
                isTraining=True, isAnomalous=True,
                doDebug=True)

preprocessor = HLpP.HostLogPreprocessor(stateCfg)
trainingStateDat:tuple[PpS, PpD] = preprocessor.GetStateData()
model = CNN(config=cnnCfg, state=trainingStateDat[0], data=trainingStateDat[1])

for file in attackDataDir.rglob("*.txt"):
    if file.is_file():        
        stateCfg.logName=f"{file.name}"
        stateCfg.logPath=file
        stateCfg.anomalyType = stateCfg.DetAnomalyType()

        preprocessor.Load(stateCfg)
        trainingStateDat = preprocessor.GetStateData()
        model.state = trainingStateDat[0]
        model.data = trainingStateDat[1]

        model.Train(10)
        
        print(f"------- RESULTS -------")
        print(f"{model.Predict()}")
        

model.Save()
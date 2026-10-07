import os
import sys

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path: sys.path.append(PARENT_DIR)

import NNUtils
import torch
import torch.nn as nn

#Use torch.relu for tensor relu NNUtils.ReLu for floats
from torch import Tensor, tensor, relu, optim, float32, long  
from NNUtils import PreprocessingData as PpD, PreprocessingState as PpS, NNConfig as NNC
from NNUtils import AnomalyType
from datetime import datetime
from pathlib import Path
from typing import Any

class ConvolNN(nn.Module):
    def __init__(self, config:NNC | str, state:PpS | str, data:PpD | str):
        super().__init__()
        self.config:NNC = None
        self.state :PpS = None
        self.data  :PpD = None

        if type(config) is not str: 
            self.config = config
        else: self.config = NNC.Load(config)
        
        if type(state) is not str:
            self.state = state
        else: self.state = PpS.Load(state)
        
        if type(data) is not str: 
            self.data = data
        else: self.data = PpD.Load(data)
        
        self.conv       = nn.Conv1d(
            in_channels = NNUtils.Vec4.Size(), #1D CNN will always use vec4 embedding
            out_channels= config.filters,
            kernel_size = config.kernelSize ) 
        
        self.pool   = nn.AdaptiveMaxPool1d(1)
        self.fcl0   = nn.Linear(config.filters, config.denseUnits)
        self.dropout= nn.Dropout(config.dropout)
        self.fcl1   = nn.Linear(config.denseUnits, len(AnomalyType))

        #self.lossFunction = nn.BCEWithLogitsLoss()  #Handles sigmoid internally
        self.lossFunction = nn.CrossEntropyLoss()
        self.lossData:list[float] = []
        self.optimizer  = optim.Adam(self.parameters(), lr=config.learningRate)

    def Debug(self) -> None:
        if self.lossData:
            print("-------- LOSS DATA --------")
            print(f"I_Loss: {self.lossData[0]}, \tF_Loss: {self.lossData[-1]}")

    #? Runs training passes and returns a list of loss values for human inspection
    def Train(self, epochs:int):
        if self.data.x is None: raise ValueError("No training data.x provided")
        if self.data.y is None: raise ValueError("No training data.y provided")

        self.train()            #Tell the pytorch NN it's in training mode
        #self.lossData.clear()

        x = tensor(self.data.x, dtype=float32)
        classIndex = self._BuildAnomalyMap(self.state.anomalyType, 
                                           self.state.isAnomalous)

        y = tensor(
            [classIndex] * len(self.data.x), 
            dtype=long)

        if self.state.doDebug:
            print(f"x_shape: {x.shape}")
            print(f"y_shape: {y.shape}")
            print(f"Expected: {self.state.anomalyType.name}")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()          #Clear leftover gradients
            output = self._Forward(x)           #Forward pass & predict
            loss = self.lossFunction(output, y) #Calculate loss
            self.lossData.append(loss.item())   #Store loss data
            loss.backward()                     #Backprop
            self.optimizer.step()               #Update weights

        if self.state.doDebug: self.Debug()

    def Predict(self) -> AnomalyType:
        if self.data.x is None: raise ValueError("No prediction data.x provided") # Put network into evaluation mode. # # This disables Dropout. 
        self.eval() 
        x = tensor(self.data.x, dtype=float32) 
        # No gradients are required during prediction. with torch.no_grad(): 
        with torch.no_grad():
            output = self._Forward(x) 
        
        # Find the class with the highest logit.
        predictions = torch.argmax(output, dim=1) 
        
        # If multiple windows were provided, each window has a 
        # prediction. For now, return the most common prediction 
        # across all windows.
        predictionIndex = torch.mode(predictions).values.item() 

        print(f"Prediction count: {torch.bincount(predictions, minlength=8)}")
        print(f"Final Prediction: {predictionIndex}")

        return self._IndexToAnomaly(predictionIndex)

    def Save(self) -> bool: 
        #? Bring this back in when implementing training saves
        saveSuccess = False
        saveTime:str = datetime.now().strftime("%H-%M-%S")
        filePath =  Path(NNUtils.NN_CFG_DIR / f"{saveTime}--{self.state.model.name}")

        saveDirPT = filePath.with_suffix(".pt")
        
        saveSuccess = self.config.Save(saveTime)
        saveSuccess = self.state.Save(saveTime)
        saveSuccess = self.data.Save(saveTime, self.state.model)

        torch.save({
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, saveDirPT )
        try: saveDirPT.exists()
        except ValueError as err:
            print(f"save error for {saveDirPT}\n{err}")
            return False
        
        return saveSuccess

    @classmethod
    def Load(cls, saveTime:str, modelType:str) -> "ConvolNN":
        #? Bring back commented code when loading training analysis data
        cfgLoadDir = f"{saveTime}--{modelType}"
        if(cfgLoadDir.endswith(".pt")): cfgLoadDir=cfgLoadDir.removesuffix(".pt")
    
        cfgPTLoadDir:Path = NNUtils.NN_CFG_DIR / f"{cfgLoadDir}.pt"
        if not cfgPTLoadDir.exists(): 
            raise ValueError(f"Cannot load data, invalid path &| filename\n{cfgPTLoadDir}...")
        
        model = cls(config=NNC.Load(cfgLoadDir), 
                    state=PpS.Load(cfgLoadDir), 
                    data=PpD.Load(cfgLoadDir))

        checkpoint = torch.load(cfgPTLoadDir, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state"])

        return model


    #?Defines network forward pass traversal
    ''' Input = [batch, windows, embeddings]
        Transposed = [batch, embeddings, windows] '''
    def _Forward(self, x:Tensor) -> Tensor:
        x = x.transpose(1,2)
        x = self.conv(x)
        x = relu(x)
        x = self.pool(x)

        ''' [batch, filters, 1] = [batch, filters] '''
        x = x.squeeze(2)
        x = self.fcl0(x)
        x = relu(x)
        x = self.dropout(x)
        x = self.fcl1(x)
        return x

    #? Use this function to save and load training statistics
    def _DatToDict(self) -> dict[str: Any]: 
        return{
            "model": self.state.model.name,

            "config": {
                "filters"   : self.config.filters,
                "kernelSize": self.config.kernelSize,
                "denseUnits": self.config.denseUnits,
                "dropout"   : self.config.dropout,
                "learningRate":self.config.learningRate },

            "training": {"lossData": self.lossData } }

    @classmethod
    def _DatFromDict(cls, dataDict:dict[str, Any]) -> "ConvolNN":
        cfg = dataDict["config"]
        newCNN = cls()

        newCNN.config = NNC(
            filters=cfg["filters"],
            kernelSize=cfg["kernelSize"],
            denseUnits=cfg["denseUnits"],
            dropout=cfg["dropout"],
            learningRate=cfg["learningRate"] )

        newCNN.lossData = dataDict["training"]["lossData"]

    @staticmethod
    def _BuildAnomalyMap(anomalyType: AnomalyType, isAnomolous:bool) -> int:
        if not isAnomolous: return 7

        try: anomalyType == AnomalyType.UNK
        except ValueError as err:
            print(f"Cannot train cnn, unknown log type...\n{err}")
            return 0

        anomalyMap = {
            "UNK"             : 0,
            "ADDUSER"         : 1,
            "HYDRAFTP"        : 2,
            "HYDRASSH"        : 3,
            "METERPRETER"     : 4,
            "JAVAMETERPRETER" : 5,
            "WEBSHELL"        : 6,
            "NORMAL"          : 7 }
        
        return anomalyMap[anomalyType.name]

    @staticmethod 
    def _IndexToAnomaly(index: int) -> AnomalyType: 
        anomalyMap = { 
            0: AnomalyType.UNK,
            1: AnomalyType.ADDUSER, 
            2: AnomalyType.HYDRAFTP, 
            3: AnomalyType.HYDRASSH, 
            4: AnomalyType.METERPRETER, 
            5: AnomalyType.JAVAMETERPRETER, 
            6: AnomalyType.WEBSHELL,
            7: AnomalyType.NORMAL } 
        
        if index == 7: return AnomalyType.NORMAL 
        if index not in anomalyMap: 
            raise ValueError( f"Invalid CNN class index: {index}" ) 
        
        return anomalyMap[index]
        
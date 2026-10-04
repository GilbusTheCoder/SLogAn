import NNUtils
import json
import torch
import torch.nn as nn

from torch import Tensor, tensor, relu, optim, dtype, float32  #Use torch.relu for tensor relu NNUtils.ReLu for floats
from NNUtils import PreprocessedData as PpD, PreprocessingState as PpS, NNConfig as NNC
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
        self.fcl1   = nn.Linear(config.denseUnits, 1)

        self.lossFunction = nn.BCEWithLogitsLoss()  #Handles sigmoid internally
        self.lossData:list[float] = []
        self.optimizer  = optim.Adam(self.parameters(), lr=config.learningRate)

    def Debug(self) -> None:
        if self.lossData:
            for data in self.lossData: print(f"{data}\n")

    #? Runs training passes and returns a list of loss values for human inspection
    def Train(self, epochs:int):
        if self.data.x is None: raise ValueError("No training data.x provided")
        if self.data.y is None: raise ValueError("No training data.y provided")

        self.train()            #Tell the pytorch NN it's in training mode
        self.lossData.clear()

        x = tensor(self.data.x, dtype=float32)
        y = tensor(
            [self.data.y] * len(self.data.x), 
            dtype=float32
        ).reshape(-1, 1) #[window_count, y_value] * len(x)

        print(x.shape)
        for epoch in range(epochs):
            self.optimizer.zero_grad()          #Clear leftover gradients
            output = self._Forward(x)           #Forward pass & predict
            loss = self.lossFunction(output, y) #Calculate loss
            self.lossData.append(loss.item())   #Store loss data
            loss.backward()                     #Backprop
            self.optimizer.step()               #Update weights

        if self.state.doDebug: self.Debug()


    def Predict(self) -> float: pass

    def Save(self) -> bool: 
        #? Bring this back in when implementing training saves
        saveSuccess = False
        saveTime:str = datetime.now().strftime("%H-%M-%S")
        filePath =  Path(NNUtils.NN_CFG_DIR / f"{saveTime}--{self.state.model.name}")

        #saveDirJSON = filePath.with_suffix(".json")
        saveDirPT = filePath.with_suffix(".pt")
        
        #saveDirJSON.mkdir(parents=True, exist_ok=True)

        #data:tuple[str: Any] = self._DatToDict()
        #with saveDirJSON.open("w", encoding="utf-8") as sfJson:
        #    json.dump(data, sfJson, indent=4)
        #try: saveDirJSON.exists()
        #except ValueError as err:
        #    print(f"save error for {saveDirJSON}\n{err}")
        #    return False

        saveSuccess = self.config.Save(saveTime)
        saveSuccess = self.state.Save(saveTime)
        saveSuccess = self.data.Save(saveTime)

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
    def Load(cls, configFileName:str, stateFileName:str, dataFileName: str) -> "ConvolNN":
        #? Bring back commented code when loading training analysis data
        cfgLoadDir = configFileName
        #if(cfgLoadDir.endswith(".json")): cfgLoadDir=cfgLoadDir.removesuffix(".json")
        if(cfgLoadDir.endswith(".pt")): cfgLoadDir=cfgLoadDir.removesuffix(".pt")

        #cfgJsonLoadDir:Path = NNUtils.NN_CFG_DIR / f"{cfgLoadDir}.json"
        #if not cfgJsonLoadDir.exists(): 
        #    raise ValueError(f"Cannot load data, invalid path &| filename\n{cfgJsonLoadDir}...")    
        cfgPTLoadDir:Path = NNUtils.NN_CFG_DIR / f"{cfgLoadDir}.json"
        if not cfgPTLoadDir.exists(): 
            raise ValueError(f"Cannot load data, invalid path &| filename\n{cfgPTLoadDir}...")

        #with cfgJsonLoadDir.open("r", encoding="utf-8") as lfJson:
        #    data = json.load(lfJson)

        model = cls(config=NNC.Load(configFileName), 
                    state=PpS.Load(stateFileName), 
                    data=PpD.Load(dataFileName))

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
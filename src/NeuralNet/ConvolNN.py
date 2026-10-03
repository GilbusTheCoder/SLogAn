import NNUtils
import torch.nn as nn

from torch import Tensor, tensor, relu, optim, dtype, float32  #Use torch.relu for tensor relu NNUtils.ReLu for floats
from NNUtils import PreprocessedData as PpD, PreprocessingState as PpS, NNConfig as NNC

class ConvolNN(nn.Module):
    def __init__(self, config:NNC, state:PpS, data:PpD):
        super().__init__()
        
        self.config:NNC = config
        self.state :PpS = state
        self.data  :PpD = data
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
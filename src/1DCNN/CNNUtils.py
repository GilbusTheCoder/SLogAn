from dataclasses import dataclass


@dataclass
class TrainingParameters:
    isTraining:bool
    isAnomolous:bool
    logName:str


def ReLu(value:float) -> float: return max(0, value)


class Neuron:
    def __init__(self, inputLen:int):
        self.weights = [0.0] * inputLen
        self.bias = 0.0

    def Output(self, inputs) -> float:
        total = self.bias
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]

        return ReLu(total)
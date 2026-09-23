import CNNPreprocessor
from CNNUtils import TrainingParameters

class CNN:
    def __init__(self, args:TrainingParameters):
        self.preprocessor = CNNPreprocessor(args)

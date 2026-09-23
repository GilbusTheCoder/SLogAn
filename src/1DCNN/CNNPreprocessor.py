''' ok so using a 1d CNN involves chunking the trace to known lengths and analysing
each chunk. This is great until you realise there's no way for a smooth-brain like
myself to identify which windows are anomolous and which are find given I only know
whether the log file in it's entirety is sus.... so we're just gonna teach the CNN
what normal logs look like, then using that data and a distribution curve we can
identify how far from normal each window is an use that to identify particularly
anomolous sections of the ADFA-LD logs 

I'd say im a genius but i still gotta build the thing....'''

import os
import sys

PARENTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENTDIR not in sys.path:
    sys.path.append(PARENTDIR)

import random
import CNNUtils
import ADFALDTranslator


MAX_CHUNK_LEN = 10
LOG_PATH = ADFALDTranslator.DATAPATH / "ADFA-LD_Logs"

#? Takes training parameters and returns convoluted windows + (metadata/labels)
class CNNPreprocessor:
    def __init__(self, args: CNNUtils.TrainingParameters):
        self.isAnomolous:bool = args.isAnomolous    

    #* Notes: 9998 = <UNK> && 9999 = <PAD>
    #TODO Add case for when log data is less than MAX_CHUNK_LEN
    def _ChunkTrace(self) -> list[list[int]]:
        rawTrace = ADFALDTranslator.TraceToInt(self.currentLog)
        windows:list[list[int]] = []
    
        for i in range(len(rawTrace) - MAX_CHUNK_LEN + 1):
            windows.append(rawTrace[i: i+MAX_CHUNK_LEN])

        for chunk in windows: print(chunk)
        return windows  #* X's


class SyscallEmbedder:
    #* Set seed for deterministic initialization
    def __init__(self, syscallCount:int, embedSize:int, seed=None):
        self.syscallCount = syscallCount
        self.embedSize = embedSize
        self.seed = random.Random(seed) #Doesn't disrupt global random generator
        self.weights = []


        for i in range(syscallCount):
            embedVector = []
            for j in range(embedSize):
                embedVector.append(random.uniform(-0.1, 0.1))

            self.weights.append(embedVector)

    def Output(self, preprocessedWindow:list[int]):
        output = []
        for syscall in preprocessedWindow: output.append(self.weights[syscall].copy())
        return output
''' Each preprocessor needs to pass two structs to the appropriate network
        1. PreprocessingState
                Contains data regarding what the preprocessor learnt about the data
                e.g.
                    - Syscall vocab
                    - Ngram vocab
                    - 

        2. PreprocessedData
                This is the actual data provided to the model with no regard for
                context and state.

''' '''
Heuristic Ideas
    - N-Gram frequency
    - Syscall frequency
    - syscall entropy / diversity '''

import os
import sys

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import NetTranslator as NT
import NNUtils

from NNUtils import PreprocessingState as PpS
from NNUtils import PreprocessedData as PpD
from dataclasses import dataclass
from pathlib import Path


#SERVER_LOG_PATH = NT.NETLOGPATH / "Foldername after Netlog/"


class NetLogPreprocessor:
    def __init__(self, config: PpS):
        if config.isHostLog: raise ValueError ("Net Preprocessor provided hostlog")
        self.isAnomolous = config.isAnomalous
        self.currentLog = self._DetLogPath(config)

    #? Use this function to return the preprocessed X and Y values for the CNN and
    #? and Clustering NN. which network the preprocessed data is being passed to should
    #? already be defined within the NNArgs
    def Preproc(self, doPrint = False) -> tuple[PpS, PpD]:
        #if doPrint: 
        #    for datum in data: print(data)
        pass

    #? Use the preprocessing config information to determine the path of the log
    def _DetLogPath(self, config: PpS) -> Path | None:
        pass
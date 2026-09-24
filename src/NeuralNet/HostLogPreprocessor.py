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

import ADFALDTranslator as AT

from NNUtils import PreprocessingState as PpS
from NNUtils import PreprocessedData as PpD
from NNUtils import ValidateConfig
from pathlib import Path

MAX_CHUNK_LEN   = 10
ADFALD_LOG_PATH = AT.HOSTLOGPATH / "ADFA-LD_Logs"


#? Takes training parameters and returns convoluted windows + (metadata/labels)
class HostLogPreprocessor:
    def __init__(self, config: PpS):
        ValidateConfig(config) #throws tantrum if you goofed an important setting 
        self.state = self._Preconfigure(config)


    #? The actual setting of data happens in here
    def Preproc(self) -> tuple[PpS, PpD]:         
        x = self._ChunkTrace()          #Get windows/trace chunks for analysis
        y = float(self.state.isAnomolous)   #0.0 if false 1.0 if true
        metadata:dict[str, any] = self._PullMetadata()

        data = PpD(x, y, metadata)
        return [self.state, data]

    #? Add any necessary data to a config and return as the preproc self.state
    def _Preconfigure(self, config:PpS) -> PpS:
        newConfig = config
        newConfig.logPath = self._DetLogPath(config)

        return newConfig


    #* Notes: 9998 = <UNK> && 9999 = <PAD>
    #TODO Add case for when log data is less than MAX_CHUNK_LEN
    def _ChunkTrace(self) -> list[list[int]]:
        rawTrace = AT.TraceToInt(self.state.logPath)
        if not rawTrace: return        

        windows:list[list[int]] = [] #* The sliding window section
        for i in range(len(rawTrace) - MAX_CHUNK_LEN + 1):
            windows.append(rawTrace[i: i+MAX_CHUNK_LEN])

        if self.state.doDebug:
            for chunk in windows: print(chunk)

        return windows  #* X's

    def _PullMetadata(self) -> dict[str, any]: pass

    #TODO Attack data wont work bc there's subfolders but it's there anyways just defunkt
    def _DetLogPath(self, config: PpS) -> Path | None:
        path = None
        if(config.isTraining):
            if(config.isAnomolous): path = ADFALD_LOG_PATH / f"Attack_Data_Master/{config.logName}"
            else: path = ADFALD_LOG_PATH / f"Training_Data_Master/{config.logName}"
        else:
            if(config.isAnomolous): path = ADFALD_LOG_PATH / f"Attack_Data_Master/{config.logName}"
            else: path = ADFALD_LOG_PATH / f"Validation_Data_Master/{config.logName}"

        return path
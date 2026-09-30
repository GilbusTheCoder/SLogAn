
from pathlib import Path
PARENT_DIR = (Path.cwd() / "src")

import ADFALDTranslator as AT
from NNUtils import PreprocessingState as PpS, PreprocessedData as PpD
from NNUtils import Vec4, Model, Format, PP_CFG_DIR

MAX_CHUNK_LEN   = 10
ADFALD_LOG_PATH = AT.HOST_LOG_PATH / "ADFA-LD_Logs"


#? Takes training parameters and returns convoluted windows + (metadata/labels)
class HostLogPreprocessor:
    def __init__(self, initConfig:PpS | str):        
        if type(initConfig) is not str:
            self.state:PpS  = self._InitState(initConfig)
            self.data:PpD   = self._InitData()
            return
        
        self.state:PpD  = self._LoadState(initConfig)
        self.data:PpD   = self._InitData()


    def GetStateData(self) -> tuple[PpS, PpD]:
        return [self.state, self.data]

    #? The actual setting of data happens in here
    def _InitData(self) -> PpD:         
        x = self._ChunkTrace()              #Get windows/trace chunks for analysis
        y = float(self.state.isAnomalous)   #0.0 if false 1.0 if true
        metadata:dict[str, any] = self._PullMetadata()

        data = PpD(x, y, metadata)
        return data

    #? Returns windows of vocabularized traces (trace stack windows translated 
    #? to the models known syscall vocab)
    #* Notes: 0 = <UNK> && 9999 = <PAD>
    def _ChunkTrace(self) -> list[list[int]]:
        rawTrace = AT.TraceToInt(self.state.logPath)
        if not rawTrace: return        
        vocabularizedTrace = self._VocabularizeTrace(rawTrace)
        windows:list[list[int]] = [] #* The sliding window section

        if len(vocabularizedTrace) < MAX_CHUNK_LEN: 
            return windows.append(self._PadWindow(vocabularizedTrace))
        
        for i in range(len(vocabularizedTrace) - MAX_CHUNK_LEN + 1):
            windows.append(vocabularizedTrace[i: i+MAX_CHUNK_LEN])        
        return windows  #* X's

    def _PadWindow(self, window:list[int]) -> list[int]: 
        newWindow = window
        while len(newWindow) < MAX_CHUNK_LEN: 
            newWindow.append(9999)
        return newWindow
        
    def _VocabularizeTrace(self, trace:list[int]) -> list[int]:
        if not self.state.vocabulary: raise ValueError("no vocabulary to reference...")
        transformedTrace:list[int] = []
        for syscall in trace:
            if syscall not in self.state.vocabulary: transformedTrace.append(0) #Unknown
            transformedTrace.append(syscall+1)

        return transformedTrace

    def _PullMetadata(self) -> dict[str, any]: 
        #TODO: This whole thing is going to be not fun....
        pass

    #? Add any necessary data to a config and return as the preproc self.state
    def _InitState(self, initialStateCfg:PpS) -> PpS:
        newConfig:PpS = initialStateCfg
        newConfig.logPath   = self._DetLogPath(initialStateCfg)
        newConfig.vocabulary= self._ConstructVocab(initialStateCfg)
        newConfig.embedding = self._ConstructEmbedding(newConfig.vocabulary)
        return newConfig

    #? This vocabulary is just syscalls with an extra unknown value.
    def _ConstructVocab(self, config: PpS) -> dict[int, str]: 
        if not AT.syscalls: raise ValueError("no syscalls available to construct vocabulary...")
        
        vocab:dict[int, str] = {0: "UNK"}
        for id, syscall in AT.syscalls.items(): 
            vocab[id+1] = syscall 

        vocab[9999] = "PAD"
        return vocab

    #? Each embed vector returned by this function is the neurons starting position
    def _ConstructEmbedding(self, vocabulary:dict[int, str]) -> dict[int, Vec4]: 
        embeddingData:dict[int, Vec4] = {}
        
        for id, vocab in vocabulary.items():
            embed = Vec4(randomize=True)
            embeddingData[id] = embed
        return embeddingData

    #TODO Attack data wont work bc there's subfolders but it's there anyways just defunkt
    def _DetLogPath(self, config: PpS) -> Path | None:
        path = None
        if(config.isTraining):
            if(config.isAnomalous): path = ADFALD_LOG_PATH / f"Attack_Data_Master/{config.logName}"
            else: path = ADFALD_LOG_PATH / f"Training_Data_Master/{config.logName}"
        else:
            if(config.isAnomalous): path = ADFALD_LOG_PATH / f"Attack_Data_Master/{config.logName}"
            else: path = ADFALD_LOG_PATH / f"Validation_Data_Master/{config.logName}"

        return path

    def _LoadState(self, cfgFileName:str) -> PpS:
        newState:PpS = PpS.Load(cfgFileName)
        return newState
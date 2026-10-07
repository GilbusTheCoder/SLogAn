
from pathlib import Path
PARENT_DIR = (Path.cwd() / "src")

import ADFALDTranslator as AT
from NNUtils import PreprocessingState as PpS, PreprocessingData as PpD
from NNUtils import Vec4, AnomalyType
from typing import Any

MAX_CHUNK_LEN   = 10
ADFALD_LOG_PATH = AT.HOST_LOG_PATH / "ADFA-LD_Logs"


#? Takes training parameters and returns convoluted windows + (metadata/labels)
class HostLogPreprocessor:
    def __init__(self, initState:PpS | str, initData:PpD | str | None = None):        
        if type(initState) is not str:
            self.state:PpS  = self._InitState(initState)
        else: self.state:PpD= self._LoadState(initData)

        if   initState.logPath is None: self.data:PpD = None
        elif initData is None:self.data:PpD = self._InitData()
        else:               self.data:PpD = self._LoadData(initData)


    def GetStateData(self) -> tuple[PpS, PpD]:
        return [self.state, self.data]

    def Save(self) -> bool:
        saveSuccess:bool = True

        if not self.state.Save(): 
            saveSuccess = False
            return saveSuccess
        if not self.data.Save(): 
            saveSuccess = False
        return saveSuccess

    def Load(self, initState:PpS | str, initData:PpD | str | None = None):
        if type(initState) is not str: self.state = self._InitState(initState)
        else: self.state = self._LoadState(initState)
        
        if initData is not None: self.data = self._LoadData(initData)
        else: self.data = self._InitData()

    #? Add any necessary data to a config and return as the preproc self.state
    def _InitState(self, initialStateCfg:PpS) -> PpS:
        newConfig:PpS = initialStateCfg
        newConfig.vocabulary= self._ConstructVocab()
        newConfig.embedding = self._ConstructEmbedding(newConfig.vocabulary)
        return newConfig

    def _LoadState(self, cfgFileName:str) -> PpS: return PpS.Load(cfgFileName)

    #? The actual setting of data happens in here
    def _InitData(self) -> PpD:
        match self.state.model.name:
            case "UNK": raise ValueError(f"Cannot init data for unknown model...")
        
            case "CONVOLUTIONAL":
                x = self._ChunkTrace()              #Get windows/trace chunks for analysis
                x = self._EmbedTrace(x)             #Apply known embeddings to chunks
                y = float(self.state.isAnomalous)   #0.0 if false 1.0 if true
                metadata:dict[str, Any] = self._PullMetadata(x)
        
            case "CLUSTERING" : pass                #TODO: This
        return PpD(x, y, metadata)

    def _LoadData(self, data:PpD | str) ->PpD:
        if type(data) is not str:
            newData:PpD = data
            newData.Debug()
            return newData
        else: return PpD.Load(data)


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

    def _EmbedTrace(self, traceWindows:list[list[int]]) -> list[list[Vec4]]:
        if not traceWindows:
            raise ValueError(f"No windows to embed...")
        
        embeddedTrace = []
        for window in traceWindows:
            embeddedChunk = []
            for trace in window:
                if not trace in self.state.embedding: 
                    embeddedChunk.append(self.state.embedding[0])
                else: embeddedChunk.append(self.state.embedding[trace])

            embeddedTrace.append(embeddedChunk)
        return embeddedTrace

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

    def _PullMetadata(self, x:object) -> dict[str, Any]: 
        metadata:dict[str, Any] = {}

        if self.state.model.name == "UNK":
            raise ValueError("Cannot pull metadata for unknown model...")

        return metadata

    
    #? This vocabulary is just syscalls with an extra unknown value.
    def _ConstructVocab(self) -> dict[int, str]: 
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
            embeddingData[id] = embed.AsList()
        return embeddingData

    def _DetLogPath(self, config: PpS) -> Path | None:
        path = None
        if(config.isTraining):
            if(config.isAnomalous): path = ADFALD_LOG_PATH / f"Attack_Data_Master/{config.logName}"
            else: path = ADFALD_LOG_PATH / f"Training_Data_Master/{config.logName}"
        else:
            anomalySet = config.logName
            anomalyTypeText = config.anomalyType.name.replace('_', '-')
            anomalySet = anomalySet.removeprefix(f"UAD-{anomalyTypeText}-")
            anomalySet = anomalySet.split('-', 1)[0]

            if(config.anomalyType is not AnomalyType.UNK and config.anomalySet > 0): 
                path = ADFALD_LOG_PATH / f"Attack_Data_Master/{config.anomalyType.name}_{anomalySet}/{config.logName}"
            else: path = ADFALD_LOG_PATH / f"Validation_Data_Master/{config.logName}"

        return path
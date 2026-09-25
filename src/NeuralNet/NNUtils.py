import os
import sys

PP_CFG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dat/cfg/Pp"))
if PP_CFG_DIR not in sys.path: 
    sys.path.append(PP_CFG_DIR)
NN_CFG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dat/cfg/NN"))
if NN_CFG_DIR not in sys.path: 
    sys.path.append(NN_CFG_DIR)

import csv
import json
import random
import numpy as np

from enum import Enum
from typing import Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

#? Helper math functions
def ReLu(x:float) -> float: return max(0, x)
def Sigmoid(x:float) -> float: return 1/(1 + np.exp(-x))

#? Vector4D class used for embedding and neuron stuff
@dataclass
class Vec4:
    def __init__(self, randomize:bool = False, seed:int | None=None, 
                 w:float=0.0, x:float=0.0, y:float=0.0, z:float=0.0):
        if randomize:
            self.Randomize(seed)
            return
        
        self.w: float = 0.00
        self.x: float = 0.00
        self.y: float = 0.00
        self.z: float = 0.00

    #? If you want a predictable starting point enter a seed
    def Randomize(self, seed:int | None = None) -> None:
        if seed: random.seed(seed)

        self.w = round(random.uniform(-1.00, 1.00), 2)
        self.x = round(random.uniform(-1.00, 1.00), 2)
        self.y = round(random.uniform(-1.00, 1.00), 2)
        self.z = round(random.uniform(-1.00, 1.00), 2)

    def AsList(self) -> list[float]: return [self.w, self.x, self.y, self.z]
    def ToConsole(self) -> None: print(f"{self.w}, {self.x}, {self.y}, {self.z}")

    def Add(self, other:Vec4) -> None:
        self.w += other.w
        self.x += other.x
        self.y += other.y
        self.z += other.z



#? A representation of a neuron.
class Neuron:
    def __init__(self, weightVecLen:int):
        self.weights = [0.0] * weightVecLen #*E.g len = 4 = [0.0, 0.0, 0.0, 0.0]
        self.bias = 0.0

    def Output(self, inputs) -> float:
        total = self.bias
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i] #*Sum(i_0j_0, i_1j_1, etc etc) + bias

        return ReLu(total)



class SaveFormat(Enum):
    JSON = 1
    TEXT = 2
    CSV  = 3

class Model(Enum):
    UNK = 0
    CONVOLVUTIONAL = 1
    CLUSTERING = 2

#TODO: Add your log formats here
class Format(Enum):
    UNK = 0
    ADFALD = 1

#? Gets passed the preprocessor, any information the preproc requires to understand
#? what it's doing goes in here. Note that note all the variables need to be set. if you
#? reference the ThomTest.py the only values I care to initialize before passing it 
#? to the preprocessor is the model, format, a few booleans and the log name. The rest
#? can and should be derived from your preprocessor. After which this preproc state struct
#? can be used as the NN state information.
@dataclass
class PreprocessingState:
    model:Model         = Model.UNK
    logFormat:Format    = Format.UNK
    logName:str | None  = None
    logPath:Path| None  = None
    
    isTraining:bool     = True
    isAnomalous:bool    = False
    isHostLog:bool      = True
    doDebug:bool        = True
    
    vocabulary:dict[int, str] | None= None
    embedding:dict[int, Vec4] | None= None

    def Debug(self):
        self._Validate()

        print("--------------  PREPROC STATE DEBUG --------------")
        print(f"\t- Model: {self.model.name}\n\t- Format: {self.logFormat.name}")
        print(f"\t- Training = {self.isTraining}")
        if(self.isTraining):print(f"\t- Log anomaly rating = {float(self.isAnomalous)}")
        if(self.isHostLog): print(f"\t- Host system log detected")
        else:               print(f"\t- Server log detected\n")

        print(f"\n-----------  Embedding & Vocabulary  ------------")
        for id, embed in self.embedding.items(): print(f"\t\t--> {id} : {self.vocabulary[id]} : {embed.AsList()}")

    def Save(self) -> None:
        for fmt in SaveFormat:        
            match(fmt):
                case SaveFormat.JSON: saveSuccess = self._SaveJSON()
                case SaveFormat.TEXT: saveSuccess = self._SaveTEXT()
                case SaveFormat.CSV:  saveSuccess = self._SaveCSV()
                case _: raise ValueError("Cannot save to unknown filetype...")

            try:   saveSuccess 
            except ValueError as err:
                print(f"Failed to save Preprocessing state to {fmt.name}...\n{err}")
                
    @classmethod
    def Load(cls, filename:str) -> "PreprocessingState":
        file = filename.lower()
        file.strip()
            
        loadedPpS = cls()
        match(filename):
            case SaveFormat.JSON: loadedPpS = cls._LoadJSON()
            case SaveFormat.TEXT: loadedPpS = cls._LoadTEXT()
            case SaveFormat.CSV:  loadedPpS = cls._LoadCSV()
            case _: raise ValueError("Cannot load unknown filetype...")

        return loadedPpS
    
    #? Throws a tantrum if you don't set something important
    def _Validate(self) -> None:
        if not self.isHostLog:           raise ValueError ("Provided host preproc with netlog...")
        if self.logName is None:         raise ValueError ("No logName Provided...")
        if self.model is Model.UNK:      raise ValueError ("No Model Provided...")
        if self.logFormat is Format.UNK: raise ValueError ("No Format Provided...")

    #* SAVING & LOADING FUNCTIONALITY
    def _SaveJSON(self) -> bool: 
        saveData:dict[str, any] = self._DatToDict()

        try: saveData
        except ValueError as err:
            print(f"Couldn't save data, no data found...\n{err}")
            return False

        jsonSaveDir:Path = PP_CFG_DIR / f"{datetime.now().strftime("%H:%M:%S")}-{self.model.name}-PpS.json"
        with jsonSaveDir.open("w", encoding="utf-8") as sfJson:
            json.dump(saveData, sfJson, indent=4)

        return True

    @classmethod
    def _LoadJSON(cls, filename:str) -> "PreprocessingState": 
        file = filename.lower()
        if(file.endswith(".json")): file.removesuffix(".json")

        loadPath:Path = PP_CFG_DIR / f"{file}.json"
        if not loadPath.exists(): 
            raise ValueError(f"Cannot load data, invalid path &| filename\n{loadPath}...")
        
        with loadPath.open("r", encoding="utf-8") as lfJson:
            loadData = json.load(lfJson)
        if not loadData: 
            raise ValueError(f"Couldn't load cfg data from {loadPath}...")
        
        return cls._DatToDict(loadData)



    def _SaveTEXT(self) -> bool: 
        textSaveDir:Path = PP_CFG_DIR / f"{datetime.now().strftime("%H:%M:%S")}-{self.model.name}-PpS.txt"

        with textSaveDir.open("w", encoding="utf-8") as sfText:
            sfText.write("[STATE]\n")
            sfText.write(f"model={self.model.name}\n")
            sfText.write(f"logFormat={self.logFormat.name}\n")
            sfText.write(f"logName={self.logName}\n")
            sfText.write(f"logPath={self.logPath}\n")
            
            sfText.write(f"\nisTraining={self.isTraining}\n")
            sfText.write(f"isAnomalous={self.isAnomalous}\n")
            sfText.write(f"isHostLog={self.isHostLog}\n")
        
            sfText.write(f"\n[VOCABULARY]\n")
            if self.vocabulary is not None:
                for id, syscall in self.vocabulary:
                    sfText.write(f"\t{id}: {syscall}\n")

            sfText.write(f"\n[EMBEDDING]\n")
            if self.embedding is not None:
                for id, embed in self.embedding:
                    values = embed.AsList()
                    sfText.write(f"\t{id}: ")

                    for value in values[:-1]: 
                        sfText.write(f"{value}, ")
                    sfText.write(f"{values[-1]}\n")

        try: textSaveDir.exists() 
        except ValueError as err:
            print(f"save error for {textSaveDir}\n{err}")
            return False
        return True

    @classmethod
    def _LoadTEXT(cls, filename:str) -> "PreprocessingState": 
        file = filename.lower()
        if(file.endswith(".txt")): file.removesuffix(".txt")

        loadPath:Path = PP_CFG_DIR / f"{file}.txt"
        if not loadPath.exists():
            raise ValueError(f"Couldn't load cfg from {loadPath}...")
        
        state = cls()
        section = None

        with loadPath.open("r", encoding="utf-8") as lfText:
            for raw_line in lfText:
                line = raw_line.strip()
                if not line: continue

                if line.startswith("[") and line.endswith("]"): 
                    section = line[1:-1]
                    continue

                match(section):
                    case "STATE": 
                        key, value = line.split("=", 1)

                        match(key):
                            case "model":     state.model = Model[value]
                            case "logFormat": state.logFormat = Format[value]
                            case "logName":   state.logName = (None if value == None else value) 
                            case "logPath":   state.logPath = (None if value == None else Path(value))
                            case "isTraining": state.isTraining = bool(value)
                            case "isAnomalous": state.isAnomalous = bool(value)
                            case "isHostLog": state.isHostLog = bool(value)
                            case "doDebug": state.doDebug = bool(value)
                            case _: raise ValueError(f"Bad State key: {key}...")

                    case "VOCABULARY": 
                        id, syscall = line.split(":", 1)
                        if state.vocabulary is None: state.vocabulary = {}
                        state.vocabulary[int(id)] = syscall

                    case "EMBEDDING": 
                        id, embed = line.split(":", 1)
                        embedValues = embed.split(",") 
                        embed = Vec4(randomize=(None if embedValues[0] == None else bool(embedValues[0])),
                                     seed=(None if embedValues[1] == None else int(embedValues[1])),
                                     w=float(embedValues[2]), x=float(embedValues[3]),
                                     y=float(embedValues[4]), z=float(embedValues[5]))
                        state.embedding[int(id)] = embed

                    case _: raise ValueError(f"Bad Section header: {section}...")
        return state
    
    def _SaveCSV(self) -> bool:  
        csvSaveDir:Path = PP_CFG_DIR / f"{datetime.now().strftime("%H:%M:%S")}-{self.model.name}-PpS.csv"

        with csvSaveDir.open("w", newline="", encoding="utf-8") as sfCsv:
            writer = csv.writer(sfCsv)

            writer.writerow(["arg", "value"])
            writer.writerow(["model", self.model.name])
            writer.writerow(["logFormat", self.logFormat.name])
            writer.writerow(["logName", self.logName])
            writer.writerow(["logPath", self.logPath])

            writer.writerow(["isTraining", self.isTraining])
            writer.writerow(["isAnomalous", self.isAnomalous])
            writer.writerow(["isHostLog", self.isHostLog])
            writer.writerow(["doDebug", self.doDebug])

            writer.writerow([
                "vocabulary", json.dumps(self.vocabulary) if self.vocabulary is not None else None])

            writer.writerow([
                "embedding", json.dumps({str(id): vec.AsList() for id, vec in self.embedding.items()})
                if self.embedding is not None
                else None ])
        
        return True
    
    @classmethod
    def _LoadCSV(cls, filename:str) -> "PreprocessingState":  
        file = filename.lower()
        if(file.endswith(".csv")): file.removesuffix(".csv")

        loadPath:Path = PP_CFG_DIR / f"{file}.csv"
        if not loadPath.exists():
            raise ValueError(f"Cannot load data, invalid path &| filename\n{loadPath}...")

        state = cls()

        with loadPath.open("r", newline="", encoding="utf-8") as lfCsv:
            reader = csv.reader(lfCsv)
            next(reader)

            for row in reader:
                key = row[0]
                arg = row[1]
                match(row[0]):
                    case "model":       state.model = Model[arg]
                    case "logFormat":   state.logFormat = Format[arg]
                    case "logName":     state.logName = arg
                    case "logPath":     Path(arg) if arg is not None else None
                    case "isTraining":  bool(arg)
                    case "isAnomalous": bool(arg)
                    case "isHostLog":   bool(arg)
                    case "doDebug":     bool(arg)
                    case "vocabulary":
                        if (arg):
                            vocabDump = json.loads(arg)
                            state.vocabulary = {
                                int(id): syscall 
                                for id, syscall in vocabDump.items() }

                        else: state.vocabulary = None

                    case "embedding":
                        if(arg):
                            embedDump = json.loads(arg)
                            state.embedding = {
                                int(id): Vec4(*embed)
                                for id, embed in embedDump.items() }

                        else: state.embedding = None 
        return state

    def _DatToDict(self) -> dict[str, Any]:
        return {
            "model": self.model.name,
            "logFormat": self.logFormat.name,
            "logName": self.logName,
            "logPath": self.logPath(
                None
                if self.logPath is None
                else(str(self.logPath))),
           
            "isTraining": self.isTraining,
            "isAnomalous": self.isAnomalous,
            "isHostLog": self.isHostLog,
            "doDebug": self.doDebug,
     
            "vocabulary": (
                None
                if self.vocabulary is None
                else{
                    str(id): syscall
                    for id, syscall in self.vocabulary.items() }),
            "embedding": (
                None
                if self.embedding is None
                else{
                    str(id): embed
                    for id, embed in self.embedding.items() }), }
    
    @classmethod
    def _DatFromDict(cls, data:dict) -> "PreprocessingState":
        loadedVocabulary:dict[int, str] = data["vocabulary"]
        if loadedVocabulary is not None:
            loadedVocabulary = {
                int(id): syscall
                for id, syscall in loadedVocabulary.items() }

        loadedEmbeddings:dict[int, Vec4] = data["embedding"]
        if loadedEmbeddings is not None:
            loadedEmbeddings = {
                int(id): Vec4(*values)
                for id, values in loadedEmbeddings.items() }

        return cls(
            model       = Model[data["model"]],
            logFormat   = Format[data["logFormat"]],
            logname     = data["logName"],
            logPath     = (
                None
                if data["logPath"] is None
                else Path(data["logPath"])),
            
            isTraining  = data["isTraining"],
            isAnomalous = data["isAnomalous"],
            isHostLog   = data["isHostLog"],
            doDebug     = data["doDebug"],

            vocabulary  = loadedVocabulary,
            embedding   = loadedEmbeddings)
    


#? The class containing our data which we pass to the models
@dataclass
class PreprocessedData:
    x: object                                #* The data we're using
    y: float | None = None                   #* The value we're trying to predict 
                                             #* (0 = norm, 1 = abnorm)
    metadata: dict[str, Any] | None   = None #* Defined by the preprocessor employed    

    def Debug(self) -> None:
        print("--------------- PREPROC DATA DEBUG  --------------")
        print(f"\t-X = {self.x}")
        print(f"\t-Y = {self.y}")

        if not self.metadata:
            print("\t-Metadata = None")
            return

        print(f"\t-Metadata")
        for id, value in self.metadata.items(): print(f"\t\t--> {id} : {value}")

    def Save(self): pass
    @classmethod
    def Load(cls, filename:str) -> "PreprocessedData": pass


    def _SaveJSON(self) -> bool:
        saveData = self._DatToDict()

        try: saveData
        except ValueError as err:
            print(f"Couldn't save data, no data found...\n{err}")
            return False

        jsonSaveDir:Path = PP_CFG_DIR / f"{datetime.now().strftime("%H:%M:%S")}-{self.model.name}-PpD.json"
        with jsonSaveDir.open("w", encoding="utf-8") as sfJson:
            json.dump(self._DatToDict(), sfJson, indent=4)
        return True

    @classmethod
    def _LoadJSON(cls, filename:str) -> "PreprocessedData":
        file = filename.lower()
        if(file.endswith(".json")): file.removesuffix(".json")

        loadPath:Path = PP_CFG_DIR / f"{file}.json"
        if not loadPath.exists():
            raise ValueError(f"Cannot load data, invalid path &| filename\n{loadPath}...")

        with loadPath.open("r", encoding="utf-8") as lfJson:
            PpD = json.load(lfJson)

        return cls._DatFromDict(PpD)

    def _SaveText(self) -> bool: 
        textSaveDir:Path = PP_CFG_DIR / f"{datetime.now().strftime("%H:%M:%S")}-{self.model.name}-PpD.txt"

        with textSaveDir.open("w", encoding="utf-8") as sfText:
            sfText.write("[X]\n")
            sfText.write(f"{json.dumps(self.x)}\n\n")
            sfText.write("[Y]\n")
            sfText.write(f"{self.y if self.y is not None else None}\n\n")
            sfText.write("[METADATA]\n")
            sfText.write(f"{json.dumps(self.metadata) if self.metadata is not None else None}\n\n")

        try: textSaveDir.exists()
        except ValueError as err:
            print(f"save error for {textSaveDir}\n{err}")
            return False
        return True

    @classmethod
    def _LoadText(cls, filename:str) -> "PreprocessedData": 
        file = filename.lower()
        if(file.endswith(".txt")): file.removesuffix(".txt")

        loadPath:Path = PP_CFG_DIR / f"{file}.txt"
        if not loadPath.exists():
            raise ValueError(f"Couldn't load cfg from {loadPath}...")

        section = None
        data = {
            "x": None,
            "y": None,
            "metadata": None }

        with loadPath.open("r", encoding="utf-8") as lfText:
            lines = [line.strip() for line in lfText]

        for line in lines:
            if not line: continue
            match(line):
                case("[X]"):        
                    section = "x"
                    continue
                case("[Y]"):        
                    section = "y"
                    continue
                case("[METADATA]"): 
                    section = "metadata"
                    continue
                case _: pass

            match(section):
                case("x"):  data["x"] = json.loads(line)
                case("y"):  data["y"] = float(line) if line else None
                case("metadata"): 
                            data["metadata"] = json.loads(line) if line else None
                case _: continue

        return cls._DatFromDict(data)


    def _SaveCSV(self) -> bool: 
        csvSaveDir:Path = PP_CFG_DIR / f"{datetime.now().strftime("%H:%M:%S")}-{self.model.name}-PpD.csv"
        
        with csvSaveDir.open("w", newline="", encoding="utf-8") as sfCsv:
            writer = csv.writer(sfCsv)
            writer.writerow(["arg", "value"])
            writer.writerow(["x", json.dumps(self.x)])
            writer.writerow(["y", None if self.y is None else self.y])
            writer.writerow(["metadata", None if self.metadata is None else json.dumps(self.metadata)])
        return True

    @classmethod
    def _LoadCSV(cls, filename:str) ->  "PreprocessedData":
        file = filename.lower()
        if(file.endswith(".csv")): file.removesuffix(".csv")
        
        loadPath:Path = PP_CFG_DIR / f"{file}.csv"
        if not loadPath.exists():
            raise ValueError(f"Cannot load data, invalid path &| filename\n{loadPath}...")
        
        loadData:dict[str, Any] = {}

        with loadPath.open("r", newline="", encoding="utf-8") as lfCsv:
            reader = csv.reader(lfCsv)
            next(reader)

            for row in reader:
                key = row[0]
                arg = row[1]

                match(row[0]):
                    case "x": loadData["x"] = json.loads(arg)
                    case "y": loadData["y"] = float(arg) if arg else None
                    case "metadata": 
                        loadData["metadata"] = json.loads(arg) if arg else None

        return cls._DatFromDict(loadData)

    def _DatToDict(self)->dict[str, Any]: 
        return {
            "x": self.x,
            "y": self.y,
            "metadata": self.metadata } 

    @classmethod
    def _DatFromDict(cls, data:dict[str, Any]) -> "PreprocessedData":
        return cls(
            x=data.get("x"),
            y=data.get("y"),
            metadata=data.get("metadata"))
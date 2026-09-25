''' 
Use this file for testing and importing whatever you need tested. We all have our 
own files to (hopefully) avoid a ton of merging that would arise when using a shared
main.py '''


import os
import sys

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path: sys.path.append(PARENT_DIR)
SIBNET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../NeuralNet"))
if SIBNET_DIR not in sys.path: sys.path.append(SIBNET_DIR)

import NeuralNet.HostLogPreprocessor as HLpP
import NeuralNet.NetLogPreprocessor as NLpP
from NeuralNet.NNUtils import PreprocessingState as PpS, NNConfig as NCC

#? Reference my ThomTest.py if you need
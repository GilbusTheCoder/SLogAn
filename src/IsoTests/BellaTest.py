''' 
Use this file for testing and importing whatever you need tested. We all have our 
own files to (hopefully) avoid a ton of merging that would arise when using a shared
main.py '''

import os
import sys

PARENTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENTDIR not in sys.path: sys.path.append(PARENTDIR)
SIBNETDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../NeuralNet"))
if SIBNETDIR not in sys.path: sys.path.append(SIBNETDIR)

import NeuralNet as NN
import NeuralNet.NNUtils as utils
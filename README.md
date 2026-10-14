# SLogAn
COS30049 NN API for analyzing logs

### Running the Models
End-to-end setup and execution instructions for both the **CNN** and **DBSCAN** models are documented directly in the workflow notebook:
**[`HostAnomalyWorkflow.ipynb`](./src/IsoTests/HostAnomalyWorkflow.ipynb)** 

Refer to the top section of the notebook for environment prerequisites, dataset validation steps, and sequential execution instructions.

### All related Analysis 
**[`HostPreprocessedAnalysis.ipynb`](./src/IsoTests/HostPreprocessedAnalysis.ipynb)** 
**[`NetPreprocessedAnalysis(DEPRECATED).ipynb`](./src/IsoTests/NetPreprocessedAnalysis(DEPRECATED).ipynb)** (Includes preprocessor for Android V1)

## Main files for each model
**[`ClusterNN.py`](./src/NeuralNet/ClusterNN.py)** 
**[`ConvoNN.py`](./src/NeuralNet/ConvoNN.py)**  

## Main preprocessor
**[`HostLogPreprocessor.py`](./src/NeuralNet/HostLogPreprocessor.py)** 
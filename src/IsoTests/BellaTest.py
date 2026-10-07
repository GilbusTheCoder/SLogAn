''' 
Use this file for testing and importing whatever you need tested. We all have our 
own files to (hopefully) avoid a ton of merging that would arise when using a shared
main.py '''


import csv
import json

config = PpS(
    logName="android.log",
    isHostLog=False,
    isAnomalous=False,  # Set True for an attack log
)

state, data = NLpP.NetLogPreprocessor(config).Preproc(doPrint=True)

feature_names = [
    state.vocabulary.get(index, f"feature_{index}")
    for index in range(len(data.x) - 2)
]
columns = feature_names + [
    "normalized_entropy",
    "event_diversity",
    "label",
    "record_count",
    "unique_event_count",
    "metadata_json",
]
row = data.x + [
    data.y,
    data.metadata["record_count"],
    data.metadata["unique_event_count"],
    json.dumps(data.metadata),
]

with open("data/net_preprocessed.csv", "w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerow(row)
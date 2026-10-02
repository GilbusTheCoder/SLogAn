'''
build_dataset.py - Reads all ADFA-LD log files and turns them into
numerical feature arrays (X_adfa.npy and y_adfa.npy) for our models.
Uses the syscall definitions from ADFALDTranslator.
'''

import os
import sys
from pathlib import Path
from collections import Counter
import numpy as np

# Point to src so we can borrow ADFALDTranslator
SRC_DIR = os.path.abspath("src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import ADFALDTranslator as AT

# Set paths to the dataset folders
LOG_ROOT = AT.HOST_LOG_PATH / "ADFA-LD_Logs"
MAX_SYSCALL_ID = max(AT.syscalls.keys()) + 1 if AT.syscalls else 340


def parse_trace_file(file_path):
    # Quick helper to read space separated numbers from a log file
    try:
        with open(file_path, "r") as f:
            tokens = f.read().split()
        return [int(t) for t in tokens if t.isdigit()]
    except Exception:
        return None


def run():
    features = []
    labels = []

    print("Building dataset from raw ADFA-LD logs...")

    # 1. Gather normal traces from Training and Validation folders (label = 0)
    normal_dirs = ["Training_Data_Master", "Validation_Data_Master"]
    for folder_name in normal_dirs:
        target_dir = LOG_ROOT / folder_name
        for log_file in target_dir.glob("*.txt"):
            trace = parse_trace_file(log_file)
            if not trace:
                continue

            # Calculate frequency of each syscall in this trace
            counts = Counter(trace)
            vec = np.zeros(MAX_SYSCALL_ID, dtype=np.float32)
            total = len(trace)

            for sc, count in counts.items():
                if sc < MAX_SYSCALL_ID:
                    vec[sc] = count / total

            features.append(vec)
            labels.append(0)

    # 2. Gather attack traces from Attack folder and subfolders (label = 1)
    attack_dir = LOG_ROOT / "Attack_Data_Master"
    for log_file in attack_dir.rglob("*.txt"):
        trace = parse_trace_file(log_file)
        if not trace:
            continue

        counts = Counter(trace)
        vec = np.zeros(MAX_SYSCALL_ID, dtype=np.float32)
        total = len(trace)

        for sc, count in counts.items():
            if sc < MAX_SYSCALL_ID:
                vec[sc] = count / total

        features.append(vec)
        labels.append(1)

    # Convert lists to NumPy arrays
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Save to disk
    np.save("X_adfa.npy", X)
    np.save("y_adfa.npy", y)

    print("\nDataset ready!")
    print(f"Saved 'X_adfa.npy' with shape: {X.shape}")
    print(f"Saved 'y_adfa.npy' with shape: {y.shape} (Normal: {np.sum(y == 0)}, Attacks: {np.sum(y == 1)})")


if __name__ == "__main__":
    run()

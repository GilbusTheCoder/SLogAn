import re
from pathlib import Path

DATAPATH = (Path.cwd() / "dat/")
SYSCALLS = (DATAPATH / "ADFA-LD_Syscall_List.txt")
syscalls = {}

def _TranslateSyscalls():
    with open(SYSCALLS, "r") as f:
        for line in f:
            match = re.match(r"#define\s+__NR_(\w+)\s(\d+)", line)

            if match:
                name = match.group(1)
                adfa_id = match.group(2)
                syscalls[adfa_id] = name

def ReadTrace(adfaLogName):
    if not syscalls: return
    adfaLog = DATAPATH / "ADFA-LD_Logs/Training_Data_Master/" / adfaLogName
    traceIDs = []

    try:
        with open(adfaLog, "r") as f:
            traceIDs = f.read().split()

        translatedTrace  = []
        for traceID in traceIDs:
            if(243 < int(traceID) < 260): sysCallName = f"custom_arch_call ({traceID})"
            else: sysCallName = syscalls.get(traceID, f"UNKNOWN({traceID})")
            
            translatedTrace.append(sysCallName)

    except: return
    for sysCall in translatedTrace: print(f"{sysCall}")

_TranslateSyscalls()
ReadTrace("UTD-0001.txt")
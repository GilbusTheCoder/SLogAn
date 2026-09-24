import re
from pathlib import Path

HOSTLOGPATH = (Path.cwd() / "dat/HostLogs")
SYSCALLS = (HOSTLOGPATH / "ADFA-LD_Syscall_List.txt")

syscalls = {}
with open(SYSCALLS, "r") as f:
    for line in f:
        match = re.match(r"#define\s+__NR_(\w+)\s(\d+)", line)
        if match:
            name = match.group(1)
            adfa_id = match.group(2)
            syscalls[adfa_id] = name


def TraceToConsole(adfaLogName) -> None:
    if not syscalls: return
    adfaLog = HOSTLOGPATH / "ADFA-LD_Logs/Training_Data_Master/" / adfaLogName
    traceIDs = []

    try:
        with open(adfaLog, "r") as l:
            traceIDs = l.read().split()

        translatedTrace  = []
        for traceID in traceIDs:
            if(243 < int(traceID) < 260): sysCallName = f"custom_arch_call ({traceID})"
            else: sysCallName = syscalls.get(traceID, f"UNKNOWN({traceID})")
            
            translatedTrace.append(sysCallName)

    except: return
    for sysCall in translatedTrace: print(f"{sysCall}")


def TraceToInt(adfaLogName) -> list[int] | None:
    if not syscalls: return
    adfaLog = HOSTLOGPATH / "ADFA-LD_Logs/Training_Data_Master/" / adfaLogName

    traceIDs = []
    try:
        with open(adfaLog, "r") as l:
            traceIDs = l.read().split()
        traceIDs = [int(trace) for trace in traceIDs]
        return traceIDs
    
    except: return None
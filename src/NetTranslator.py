''' Put translation and log parsing utility functions in here to keep the 
    preprocessors clean and readable. It's already linked up and everything should
    be accessible from the NetPreprocessor by default. '''

from pathlib import Path

NETLOGPATH = (Path.cwd() / "dat/NetLogs")
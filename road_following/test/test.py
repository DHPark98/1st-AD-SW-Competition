import os
import sys
from pathlib import Path
GLOBAL_PATH = str(Path(os.getcwd()).parent.parent)
sys.path.append(GLOBAL_PATH)
from Global.LogManager import Log

log = Log()

log.debug()
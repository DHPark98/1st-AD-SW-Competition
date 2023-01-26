import os
import sys
from pathlib import Path
PATH = str(Path(os.path.dirname(__file__)).parent)
sys.path.append(PATH)

from Devices.Lidar import LidarModule


lidar_module = LidarModule()

lidar_module.iter_scans()
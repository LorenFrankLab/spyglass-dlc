import numpy as np
import pandas as pd
import math
import datajoint as dj
from datetime import datetime
import deeplabcut
import pynwb
import os
import sys
import glob
import ruamel.yaml as yaml
from typing import List, Dict, OrderedDict
from pathlib import Path
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from dgramling_dlc_pose_estimation import DLCPoseEstimation

schema = dj.schema('dgramling_dlc_position')

@schema
class DLCSmoothInterpParams(self, key):
    definition="""
    -> DLCPoseEstimation
    -> VideoFile
    """
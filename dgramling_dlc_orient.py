import numpy as np
import pandas as pd
import math
import datajoint as dj
from datetime import datetime
import pynwb
import os
import sys
from itertools import groupby
from operator import itemgetter
import bottleneck as bn
from typing import List, Dict, OrderedDict
from pathlib import Path
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_behav import RawPosition
from .dgramling_dlc_pose_estimation import DLCPoseEstimation
from .dgramling_dlc_project import BodyPart

schema = dj.schema('dgramling_dlc_orient')
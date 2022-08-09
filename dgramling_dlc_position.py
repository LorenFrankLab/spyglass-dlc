from lib2to3.pygram import pattern_symbols
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
class DLCSmoothInterpParams(dj.Manual):
    definition="""
    dlc_si_params_name : varchar(40) # concise name for parameter set
    ---
    params : longblob # dictionary to hold parameters
    """
    @classmethod
    def insert_params(cls, key, params:dict=None, skip_duplicates=True):
        pass

@schema
class DLCSmoothInterpSelection(dj.Manual):
    definition="""
    -> DLCPoseEstimation
    -> VideoFile # Not sure this is necessary given that video file is linked to DLCPoseEstimatino
    -> DLCSmoothInterpParams
    ---

    """

@schema
class DLCSmoothInterp(dj.Computed):
    definition="""
    -> DLCSmoothInterpSelection
    ---
    -> AnalysisNwbfile
    dlc_position_object_id : varchar(80)
    """
    
    def make(self, key):
        # Get labels to smooth from Parameters table
        params = (DLCSmoothInterpParams)() & key).fetch1('dlc_si_params_name')
        labels = params['labels']
        # Get DLC output dataframe
        dlc_df = (DLCPoseEstimation() & key).fetch1_dataframe()[0]
        interp_df = dlc_df.copy()
        interp_df = self.interp_pos(interp_df, params['interp_params'])
        self.smooth_pos()
        
        # Get positions for the specified labels


        # TODO: conceptualize how to take centroid of n-points
    
    def interp_pos(self, ...):
        pass

    def smooth_pos(self, ...):
        pass
    

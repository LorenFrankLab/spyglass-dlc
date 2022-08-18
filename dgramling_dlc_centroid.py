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
from .dgramling_dlc_position import DLCSmoothInterp
from .dgramling_dlc_project import BodyPart

schema = dj.schema('dgramling_dlc_centroid')

@schema
class DLCSmoothInterpCohortSelection(dj.Manual):
    """
    """
    definition="""
    dlc_si_cohort_selection_name : varchar(120):
    ---
    -> DLCSmoothInterp
    bodyparts : blob

    """


@schema
class DLCSmoothInterpCohort(dj.Computed):
    """
    """
    definition="""
    -> DLCCentroidCohortSelection
    """

    class BodyPart(dj.Part):
        definition="""
        -> DLCCentroidCohortSelection
        -> DLCSmoothInterp
        ---
        -> AnalysisNwbfile
        dlc_position_object_id : varchar(80)
        """
    
    def make(self, key):
        # from Jen Guidera
        self.insert1(key)
        cohort_selection = (DLCSmoothInterpCohortSelection & key).fetch1()
        table_entries = (DLCSmoothInterp & cohort_selection).fetch()
        table_column_names = list(table_entries.dtype.fields.keys())
        for table_entry in table_entries:
            entry_key = {**{k: v for k, v in zip(table_column_names, table_entry)},
                         **key}
            DLCSmoothInterpCohort.BodyPart.insert1(entry_key, skip_duplicates=True)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'),
                         *attrs, **kwargs)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]['dlc_position'].set_index('time')


@schema
class DLCCentroidParams(dj.Manual):
    """Parameters for calculating the centroid
    """
    definition="""
    dlc_centroid_params_name: varchar(80) # name for this set of parameters
    ---
    max_separation = 9.0  : float   # max distance (in cm) between head LEDs
    max_speed = 300.0     : float   # max speed (in cm / s) of animal
    params: longblob
    """

    @classmethod
    def insert_params(cls, params):
        pass


@schema
class DLCCentroidSelection(dj.Manual):
    """
    """
    definition="""
    -> DLCSmoothInterpCohort
    -> DLCCentroidParams
    ---
    """

@schema
class DLCCentroid(dj.Computed):
    """
    """
    definition="""
    -> DLCCentroidSelection
    ---
    -> AnalysisNwbfile
    dlc_centroid_object_id : varchar(80)
    """

    def make(self, key):
        key['analysis_file_name'] = AnalysisNwbfile().create(
                                    key['nwb_file_name'])
        # Get labels to smooth from Parameters table
        cohort_entries = (DLCSmoothInterpCohort.BodyPart & key)
        part_dfs = []
        for bodypart in cohort_entries.fetch('bodyparts'):
            part_dfs.append((DLCSmoothInterpCohort.BodyPart & {**key, **{
                "bodypart": bodypart}}).fetch1_dataframe())
        params = (DLCCentroidParams() & key).fetch1()
        raw_position = (RawPosition() &
                        {'nwb_file_name': key['nwb_file_name'],
                         'interval_list_name': key['interval_list_name']
                         }).fetch_nwb()[0]
        meters_to_pixels = raw_position['raw_position'].conversion
        # centroid_bodyparts = (DLCCentroidCohortSelection.BodyPart & key).fetch()
        query = DLCSmoothInterp() & key
        bodyparts = (BodyPart & key).fetch('bodypart')
        bodypart_dfs = {
            entry['bodypart']: entry.fetch1_dataframe() for entry in query}
        concat_df = pd.concat(
            list(bodypart_dfs.values()),
            axis=1,
            keys=list(bodypart_dfs.keys()))
        centroid_func = _key_to_func_dict(params['centroid_method'])
        centroid = centroid_func(concat_df, **params['centroid_params'])



def four_led_centroid(pos_df, **params):
    
    centroid = None
    return centroid

def two_pt_centroid():
    raise NotImplementedError

def one_pt_centroid():
    raise NotImplementedError


_key_to_func_dict = {
    'four_led_centroid': four_led_centroid,
    'two_pt_centroid': two_pt_centroid,
    'one_pt_centroid': one_pt_centroid,
}
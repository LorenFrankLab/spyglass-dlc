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
from .dgramling_dlc_position import DLCSmoothInterp
from .dgramling_dlc_project import BodyPart

schema = dj.schema("dgramling_dlc_cohort")


@schema
class DLCSmoothInterpCohortSelection(dj.Manual):
    """ """

    definition = """
    dlc_si_cohort_selection_name : varchar(120)
    ---
    -> DLCSmoothInterp
    bodyparts : blob

    """


@schema
class DLCSmoothInterpCohort(dj.Computed):
    """ """

    definition = """
    -> DLCSmoothInterpCohortSelection
    """

    class BodyPart(dj.Part):
        definition = """
        -> DLCSmoothInterpCohortSelection
        -> DLCSmoothInterp
        ---
        -> AnalysisNwbfile
        dlc_position_object_id : varchar(80)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

        def fetch1_dataframe(self):
            return self.fetch_nwb()[0]["dlc_position"].set_index("time")

    def make(self, key):
        # from Jen Guidera
        self.insert1(key)
        cohort_selection = (DLCSmoothInterpCohortSelection & key).fetch1()
        table_entries = (DLCSmoothInterp & cohort_selection).fetch()
        table_column_names = list(table_entries.dtype.fields.keys())
        for table_entry in table_entries:
            entry_key = {
                **{k: v for k, v in zip(table_column_names, table_entry)},
                **key,
            }
            DLCSmoothInterpCohort.BodyPart.insert1(entry_key, skip_duplicates=True)

import numpy as np
import pandas as pd
import datajoint as dj
from typing import List, Dict, OrderedDict
from pathlib import Path
from position_tools.core import gaussian_smooth
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from .dgramling_dlc_cohort import DLCSmoothInterpCohort
from .dgramling_dlc_project import BodyPart

schema = dj.schema("dgramling_dlc_orient")


@schema
class DLCOrientationParams(dj.Manual):
    """Parameters for calculating the centroid"""

    definition = """
    dlc_orientation_params_name: varchar(80) # name for this set of parameters
    ---
    params: longblob
    """

    @classmethod
    def insert_params(cls, params_name: str, params: dict):
        cls.insert1({"dlc_orientation_params_name": params_name, "params": params})

    @classmethod
    def insert_default(cls):
        params = {
            "orient_method": "red_bisector",
            "led1": "redLED_L",
            "led2": "redLED_R",
            "led3": "redLED_C",
        }
        cls.insert1({"dlc_orientation_params_name": "default", "params": params})


@schema
class DLCOrientationSelection(dj.Manual):
    """ """

    definition = """
    -> DLCSmoothInterpCohort
    -> DLCOrientationParams
    ---
    """


@schema
class DLCOrientation(dj.Computed):
    """ """

    definition = """
    -> DLCOrientationSelection
    ---
    -> AnalysisNwbfile
    dlc_orientation_object_id : varchar(80)
    """

    def make(self, key):
        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
        # Get labels to smooth from Parameters table
        cohort_entries = DLCSmoothInterpCohort.BodyPart & key
        pos_df = pd.concat(
            {
                bodypart: (
                    DLCSmoothInterpCohort.BodyPart & {**key, **{"bodypart": bodypart}}
                ).fetch1_dataframe()
                for bodypart in cohort_entries.fetch("bodyparts")
            }
        )
        params = (DLCOrientationParams() & key).fetch1("params")
        orientation_smoothing_std_dev = params.pop(
            "orientation_smoothing_std_dev", None
        )
        dt = np.median(np.diff(pos_df.index.to_numpy()))
        sampling_rate = 1 / dt
        orient_func = _key_to_func_dict[params["orient_method"]]
        orientation = orient_func(pos_df, **params)
        # Smooth orientation
        is_nan = np.isnan(orientation)
        # Unwrap orientation before smoothing
        orientation[~is_nan] = np.unwrap(orientation[~is_nan])
        orientation[~is_nan] = gaussian_smooth(
            orientation[~is_nan],
            orientation_smoothing_std_dev,
            sampling_rate,
            axis=0,
            truncate=8,
        )
        # convert back to between -pi and pi
        orientation[~is_nan] = np.angle(np.exp(1j * orientation[~is_nan]))
        final_df = pd.DataFrame(
            orientation, columns=["orientation"], index=pos_df.index
        )
        nwb_analysis_file = AnalysisNwbfile()
        key["dlc_orientation_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=final_df,
        )
        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

        def fetch1_dataframe(self):
            return self.fetch_nwb()[0]["dlc_orientation"].set_index("time")


def two_pt_head_orientation(pos_df: pd.DataFrame, **params):
    """Determines orientation based on vector between two points"""
    BP1 = params.pop("bodypart1", None)
    BP2 = params.pop("bodypart2", None)
    orientation = np.arctan2(
        (pos_df[BP1]["y"] - pos_df[BP2]["y"]),
        (pos_df[BP1]["x"] - pos_df[BP2]["x"]),
    )
    return orientation


def red_led_bisector_orientation(pos_df: pd.DataFrame, **params):
    """Determines orientation based on 2 equally-spaced identifiers
    that are assumed to be perpendicular to the orientation direction.
    A third object is needed to determine forward/backward
    """
    LED1 = params.pop("led1", None)
    LED2 = params.pop("led2", None)
    LED3 = params.pop("led3", None)
    orientation = []
    for index, row in pos_df.iterrows():
        x_vec = row[LED1]["x"] - row[LED2]["x"]
        y_vec = row[LED1]["y"] - row[LED2]["y"]
        if y_vec == 0:
            if (row[LED3]["y"] > row[LED1]["y"]) & (row[LED3]["y"] > row[LED2]["y"]):
                orientation.append(np.pi / 2)
            elif (row[LED3]["y"] < row[LED1]["y"]) & (row[LED3]["y"] < row[LED2]["y"]):
                orientation.append(-(np.pi / 2))
            else:
                raise Exception("Cannot determine head direction from bisector")
        else:
            length = np.sqrt(y_vec * y_vec + x_vec * x_vec)
            norm = np.array([-y_vec / length, x_vec / length])
            orientation.append(np.arctan2(norm[1], norm[0]))
        if index + 1 == len(pos_df):
            break
    return orientation


# Add new functions for orientation calculation here

_key_to_func_dict = {
    "red_green_orientation": two_pt_head_orientation,
    "red_led_bisector": red_led_bisector_orientation,
}

import numpy as np
import pandas as pd
import datajoint as dj
import pynwb
import pynwb.behavior
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_behav import RawPosition
from spyglass.common.common_interval import IntervalList
from .dgramling_dlc_centroid import DLCCentroid
from .dgramling_dlc_orient import DLCOrientation

schema = dj.schema("dgramling_dlc_selection")


@schema
class DLCPosSelection(dj.Manual):
    definition = """
    -> DLCCentroid.proj(dlc_si_cohort_centroid='dlc_si_cohort_selection_name')
    -> DLCOrientation.proj(dlc_si_cohort_orientation='dlc_si_cohort_selection_name')
    """


"""
Not sure this will work... would ideally combine centroid and orientation into single entry with single dataframe,
but may be difficult to resolve primary keys if cohorts different
"""


@schema
class DLCPos(dj.Computed):
    definition = """
    -> DLCPosSelection
    ---
    -> AnalysisNwbfile
    position_object_id      : varchar(80)
    orientation_object_id   : varchar(80)
    velocity_object_id      : varchar(80)
    """

    def make(self, key):
        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
        centroid_df = (DLCCentroid & key).fetch1_dataframe()
        orientation_df = (DLCOrientation & key).fetch1_dataframe()
        spatial_series = (RawPosition & key).fetch_nwb()[0]["raw_position"]
        final_df = centroid_df.join([orientation_df])
        position = pynwb.behavior.Position()
        orientation = pynwb.behavior.CompassDirection()
        velocity = pynwb.behavior.BehavioralTimeSeries()
        METERS_PER_CM = 0.01
        idx = pd.IndexSlice
        position.create_spatial_series(
            name="position",
            timestamps=final_df.index.to_numpy(),
            conversion=METERS_PER_CM,
            data=final_df.loc[:, idx[("position_x", "position_y")]],
            reference_frame=spatial_series.reference_frame,
            comments=spatial_series.comments,
            description="x_position, y_position",
        )

        orientation.create_spatial_series(
            name="orientation",
            timestamps=final_df.index.to_numpy(),
            conversion=1.0,
            data=final_df.loc[:, idx[("orientation")]],
            reference_frame=spatial_series.reference_frame,
            comments=spatial_series.comments,
            description="orientation",
        )

        velocity.create_timeseries(
            name="velocity",
            timestamps=final_df.index.to_numpy(),
            conversion=METERS_PER_CM,
            unit="m/s",
            data=final_df.loc[:, idx[("velocity_x", "velocity_y", "speed")]],
            comments=spatial_series.comments,
            description="x_velocity, y_velocity, speed",
        )

        nwb_analysis_file = AnalysisNwbfile()

        key["position_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], position
        )
        key["orientation_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], orientation
        )
        key["velocity_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], velocity
        )

        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])

        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        nwb_data = self.fetch_nwb()[0]
        index = pd.Index(
            np.asarray(nwb_data["position"].get_spatial_series().timestamps),
            name="time",
        )
        COLUMNS = [
            "position_x",
            "position_y",
            "orientation",
            "velocity_x",
            "velocity_y",
            "speed",
        ]
        return pd.DataFrame(
            np.concatenate(
                (
                    np.asarray(nwb_data["position"].get_spatial_series().data),
                    np.asarray(nwb_data["orientation"].get_spatial_series().data)[
                        :, np.newaxis
                    ],
                    np.asarray(nwb_data["velocity"].time_series["velocity"].data),
                ),
                axis=1,
            ),
            columns=COLUMNS,
            index=index,
        )

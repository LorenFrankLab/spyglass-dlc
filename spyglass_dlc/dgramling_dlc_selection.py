import numpy as np
import pandas as pd
import datajoint as dj
import pynwb
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
    -> DLCCentroid.proj(dlc_si_cohort_centroid='dlc_si_cohort_selection_name', centroid_analysis_file_name='analysis_file_name')
    -> DLCOrientation.proj(dlc_si_cohort_orientation='dlc_si_cohort_selection_name', orientation_analysis_file_name='analysis_file_name')
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
        position_nwb_data = (DLCCentroid & key).fetch_nwb()[0]
        orientation_nwb_data = (DLCOrientation & key).fetch_nwb()[0]
        position_object = position_nwb_data["dlc_position"].spatial_series["position"]
        velocity_object = position_nwb_data["dlc_velocity"].time_series["velocity"]
        orientation_object = orientation_nwb_data["dlc_orientation"].spatial_series[
            "orientation"
        ]
        position = pynwb.behavior.Position()
        orientation = pynwb.behavior.CompassDirection()
        velocity = pynwb.behavior.BehavioralTimeSeries()
        position.create_spatial_series(
            name=position_object.name,
            timestamps=np.asarray(position_object.timestamps),
            conversion=position_object.conversion,
            data=np.asarray(position_object.data),
            reference_frame=position_object.reference_frame,
            comments=position_object.comments,
            description=position_object.description,
        )
        orientation.create_spatial_series(
            name=orientation_object.name,
            timestamps=np.asarray(orientation_object.timestamps),
            conversion=orientation_object.conversion,
            data=np.asarray(orientation_object.data),
            reference_frame=orientation_object.reference_frame,
            comments=orientation_object.comments,
            description=orientation_object.description,
        )
        velocity.create_timeseries(
            name=velocity_object.name,
            timestamps=np.asarray(velocity_object.timestamps),
            conversion=velocity_object.conversion,
            unit=velocity_object.unit,
            data=np.asarray(velocity_object.data),
            comments=velocity_object.comments,
            description=velocity_object.description,
        )
        # Add to Analysis NWB file
        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
        nwb_analysis_file = AnalysisNwbfile()

        key["orientation_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], orientation
        )
        key["position_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], position
        )
        key["velocity_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], velocity
        )

        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )

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

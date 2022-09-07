import datajoint as dj
import pandas as pd
import numpy as np
import pynwb
import pynwb.behavior
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_interval import IntervalList
from .dgramling_dlc_selection import DLCPos
from .dgramling_trodes_position import TrodesPos

schema = dj.schema("dgramling_position")


@schema
class PosSelect(dj.Manual):
    """ """

    # TODO: I think the IntervalList dependency should be replaced by a table further downstream
    definition = """
    -> IntervalList
    ---
    source : enum("DLC", "Trodes")
    """

    class DLCPos(dj.Part):
        """ """

        definition = """
        -> PosSelect
        -> DLCPos
        ---
        -> AnalysisNwbfile
        position_object_id : varchar(80)
        orientation_object_id : varchar(80)
        velocity_object_id : varchar(80)
        """

    class TrodesPos(dj.Part):
        """ """

        definition = """
        -> PosSelect
        -> TrodesPos
        ---
        -> AnalysisNwbfile
        position_object_id : varchar(80)
        orientation_object_id : varchar(80)
        velocity_object_id : varchar(80)
        """


@schema
class IntervalPositionInfoSelection(dj.Manual):
    """ """

    definition = """
    -> PosSelect
    ---
    """


@schema
class IntervalPositionInfo(dj.Computed):
    """ """

    definition = """
    -> IntervalPositionInfoSelection
    ---
    -> AnalysisNwbfile
    position_object_id : varchar(80)
    orientation_object_id : varchar(80)
    velocity_object_id : varchar(80)
    """

    def make(self, key):
        source = (PosSelect & key).fetch1("source")
        table_source = f"{source}Pos"
        SourceTable = getattr(PosSelect, table_source)
        (
            key["analysis_file_name"],
            key["position_object_id"],
            key["orientation_object_id"],
            key["velocity_object_id"],
        ) = (SourceTable & key).fetch1(
            "analysis_file_name",
            "position_object_id",
            "orientation_object_id",
            "velocity_object_id",
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

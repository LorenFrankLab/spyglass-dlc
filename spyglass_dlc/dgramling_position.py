import datajoint as dj
import pandas as pd
import numpy as np
from typing import Dict
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_interval import IntervalList
from .dgramling_dlc_selection import DLCPos
from .dgramling_trodes_position import TrodesPos

schema = dj.schema("dgramling_position")
# TODO: automatically insert from TrodesPos or DLCPos into PosSource Part tables at the end of make


@schema
class PosSelect(dj.Manual):
    """
    Table to specify which entry from upstream pipeline should be added to PosSource
    Allows for multiple entries per epoch per source with incrementing position_id key
    If specifying DLC as upstream, set dlc_params foreign key with dict of keys necessary
    to query DLCPos
    """

    # This would limit the user to only one entry per interval in IntervalPosInfo
    definition = """
    -> IntervalList
    source: enum("DLC", "Trodes")
    position_id: int
    ---
    dlc_params = NULL: longblob     # dictionary with primary keys of upstream DLC entries
    """

    def insert1(self, key, **kwargs):
        # TODO: not sure this logic with if/else makes sense...
        position_id = key.get("position_id", None)
        if position_id is None:
            key["position_id"] = (
                dj.U().aggr(self, n="max(position_id)").fetch1("n") or 0
            ) + 1
        else:
            id = (self & key).fetch("position_id")
            if len(id) > 0:
                position_id = max(id) + 1
            else:
                position_id = max(0, position_id)
            key["position_id"].update(position_id)
        super().insert1(key, **kwargs)


@schema
class PosSource(dj.Manual):
    """
    Table to identify source of Position Information from upstream options
    (e.g. DLC, Trodes, etc...) To add another upstream option, a new Part table
    should be added in the same syntax as DLCPos and TrodesPos and
    PosSelect source header should be modified to include the name.
    """

    definition = """
    -> IntervalList
    source: enum("DLC", "Trodes")
    position_id: int
    ---
    """

    class DLCPos(dj.Part):
        """
        Table to pass-through upstream DLC Pose Estimation information
        """

        definition = """
        -> PosSource
        -> DLCPos
        ---
        -> AnalysisNwbfile
        position_object_id : varchar(80)
        orientation_object_id : varchar(80)
        velocity_object_id : varchar(80)
        """

    class TrodesPos(dj.Part):
        """
        Table to pass-through upstream Trodes Position Tracking information
        """

        definition = """
        -> PosSource
        -> TrodesPos
        ---
        -> AnalysisNwbfile
        position_object_id : varchar(80)
        orientation_object_id : varchar(80)
        velocity_object_id : varchar(80)
        """

    def insert1(self, key, params: Dict = None, **kwargs):
        position_id = key.get("position_id", None)
        if position_id is None:
            key["position_id"] = (
                dj.U().aggr(self, n="max(position_id)").fetch1("n") or 0
            ) + 1
        else:
            id = (self & key).fetch("position_id")
            if len(id) > 0:
                position_id = max(id) + 1
            else:
                position_id = max(0, position_id)
            key["position_id"] = position_id
        super().insert1(key, **kwargs)
        source = key["source"]
        part_table = getattr(self, f"{source}Pos")
        table_query = (
            dj.FreeTable(dj.conn(), full_table_name=part_table.parents()[1]) & params
        )
        (
            analysis_file_name,
            position_object_id,
            orientation_object_id,
            velocity_object_id,
        ) = table_query.fetch1(
            "analysis_file_name",
            "position_object_id",
            "orientation_object_id",
            "velocity_object_id",
        )
        part_table.insert1(
            {
                **key,
                "analysis_file_name": analysis_file_name,
                "position_object_id": position_object_id,
                "orientation_object_id": orientation_object_id,
                "velocity_object_id": velocity_object_id,
                **params,
            },
        )


@schema
class IntervalPositionInfoSelection(dj.Manual):
    """
    Table to specify which upstream PosSelect entry to populate IntervalPositionInfo
    """

    definition = """
    -> PosSource
    ---
    """


@schema
class IntervalPositionInfo(dj.Computed):
    """
    Holds position information in a singluar location for an
    arbitrary number of upstream position processing options
    """

    definition = """
    -> IntervalPositionInfoSelection
    ---
    -> AnalysisNwbfile
    position_object_id : varchar(80)
    orientation_object_id : varchar(80)
    velocity_object_id : varchar(80)
    """

    def make(self, key):
        source = (PosSource & key).fetch1("source")
        table_source = f"{source}Pos"
        SourceTable = getattr(PosSource, table_source)
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
            "video_frame_ind",
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
                    np.asarray(
                        nwb_data["velocity"].time_series["video_frame_ind"].data,
                        dtype=int,
                    )[:, np.newaxis],
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

import datajoint as dj
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_interval import IntervalList
from .dgramling_dlc_position import DLCSmoothInterp
from .dgramling_dlc_project import BodyPart

schema = dj.schema("dgramling_dlc_cohort")


@schema
class DLCSmoothInterpCohortSelection(dj.Manual):
    """
    Table to specify which combination of bodyparts from DLCSmoothInterp
    get combined into a cohort
    """

    # TODO: try to make a strict dependence on DLCSmoothInterpSelection
    definition = """
    -> IntervalList
    dlc_si_cohort_selection_name : varchar(120)
    ---
    epoch                   : int           # the session epoch for this task and apparatus(1 based)
    video_file_num          : int
    dlc_model_name          : varchar(64)   # User-friendly model name
    dlc_model_params_name   : varchar(40)   
    dlc_si_params_name      : varchar(80)   # descriptive name of this interval list
    bodyparts               : blob          # List of bodyparts to include in cohort
    """


@schema
class DLCSmoothInterpCohort(dj.Computed):
    """
    Table to combine multiple bodyparts from DLCSmoothInterp
    to enable centroid/orientation calculations
    """

    # Need to ensure that nwb_file_name/epoch/interval list name endure as primary keys
    definition = """
    -> DLCSmoothInterpCohortSelection
    ---
    """

    class BodyPart(dj.Part):
        definition = """
        -> DLCSmoothInterpCohortSelection
        -> DLCSmoothInterp
        ---
        -> AnalysisNwbfile
        dlc_smooth_interp_object_id : varchar(80)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
            )

        def fetch1_dataframe(self):
            return self.fetch_nwb()[0]["dlc_smooth_interp"].set_index("time")

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

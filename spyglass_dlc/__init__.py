from .dgramling_dlc_project import DLCProject, BodyPart
from .dgramling_dlc_training import (
    DLCModelTrainingParams,
    DLCModelTrainingSelection,
    DLCModelTraining,
)
from .dgramling_dlc_model import (
    DLCModelSource,
    DLCModelInput,
    DLCModelParams,
    DLCModelSelection,
    DLCModel,
    DLCModelEvaluation,
)
from .dgramling_dlc_pose_estimation import DLCPoseEstimationSelection, DLCPoseEstimation
from .dgramling_dlc_position import (
    DLCSmoothInterpParams,
    DLCSmoothInterpSelection,
    DLCSmoothInterp,
)
from .dgramling_dlc_cohort import DLCSmoothInterpCohortSelection, DLCSmoothInterpCohort
from .dgramling_dlc_centroid import DLCCentroidParams, DLCCentroidSelection, DLCCentroid
from .dgramling_dlc_orient import (
    DLCOrientationParams,
    DLCOrientationSelection,
    DLCOrientation,
)
from .dlc_utils import (
    get_dlc_root_data_dir,
    get_dlc_processed_data_dir,
    find_full_path,
    find_root_directory,
    _convert_mp4,
)


def schemas():
    return _schemas


_schemas = [
    "dgramling_dlc_project",
    "dgramling_dlc_training",
    "dgramling_dlc_model",
    "dgramling_dlc_pose_estimation",
    "dgramling_dlc_position",
    "dgramling_dlc_cohort",
    "dgramling_dlc_centroid",
    "dgramling_dlc_orient",
]

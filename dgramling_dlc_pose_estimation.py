from socket import if_indextoname
from urllib.parse import non_hierarchical
import numpy as np
import pandas as pd
import math
import datajoint as dj
import deeplabcut
import pynwb
import os
import sys
import glob
import ruamel.yaml as yaml
from typing import List, Dict, OrderedDict
from pathlib import Path
from dlc_utils import find_full_path, find_root_directory
from spyglass.common.common_lab import LabTeam
from dgramling_dlc_project import BodyPart
from dgramling_dlc_model import DLCModel
from spyglass.common.common_behav import VideoFile

schema = dj.schema('dgramling_dlc_pose_estimation')

@schema
class DLCPoseEstimationSelection(dj.Manual):
    definition = """
    -> VideoFile                           # Session -> Recording + File part table
    -> DLCModel                                    # Must specify a DLC project_path

    ---
    task_mode='load' : enum('load', 'trigger')  # load results or trigger computation
    pose_estimation_output_dir='': varchar(255) # output dir relative to the root dir
    pose_estimation_params=null  : longblob     # analyze_videos params, if not default
    """

    @classmethod
    def infer_output_dir(cls, key, relative=False, mkdir=False):
        """Return the expected pose_estimation_output_dir.

        With spaces in model name are replaced with hyphens.
        Based on convention: / video_dir / Device_{}_Recording_{}_Model_{}

        Parameters
        ----------
        key: DataJoint key specifying a pairing of VideoRecording and Model.
        relative (bool): Report directory relative to get_dlc_processed_data_dir().
        mkdir (bool): Default False. Make directory if it doesn't exist.
        """
        # Get video_filepath
        video_info = (VideoFile() &
                      {'nwb_file_name': key['nwb_file_name'],
                       'epoch': epoch}).fetch1()
        io = pynwb.NWBHDF5IO('/stelmo/nwb/raw/' +
                             video_info['nwb_file_name'], 'r')
        nwb_file = io.read()
        nwb_video = nwb_file.objects[video_info['video_file_object_id']]
        video_filepath = nwb_video.external_file.file.filename
        video_filepath = find_full_path(
            get_dlc_root_data_dir(),
            (VideoFile & key)
        )
        root_dir = find_root_directory(get_dlc_root_data_dir(), video_filepath.parent)
        recording_key = VideoFile & key
        if get_dlc_processed_data_dir():
            processed_dir = Path(get_dlc_processed_data_dir())
        else:  # if processed not provided, default to where video is
            processed_dir = root_dir

        output_dir = (
            processed_dir
            / video_filepath.parent.relative_to(root_dir)
            / (
                f'video_filename_model_'
                + key["model_name"].replace(" ", "-")
            )
        )
        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def insert_estimation_task(
        cls,
        key,
        task_mode="trigger",
        params: dict = None,
        relative=True,
        mkdir=True,
        skip_duplicates=False,
    ):
        """Insert PoseEstimationTask in inferred output dir.

        Based on the convention / video_dir / device_{}_recording_{}_model_{}

        Parameters
        ----------
        key: DataJoint key specifying a pairing of VideoRecording and Model.
        task_mode (bool): Default 'trigger' computation. Or 'load' existing results.
        params (dict): Optional. Parameters passed to DLC's analyze_videos:
            videotype, gputouse, save_as_csv, batchsize, cropping, TFGPUinference,
            dynamic, robust_nframes, allow_growth, use_shelve
        relative (bool): Report directory relative to get_dlc_processed_data_dir().
        mkdir (bool): Default False. Make directory if it doesn't exist.
        """
        output_dir = cls.infer_output_dir(key, relative=relative, mkdir=mkdir)

        cls.insert1(
            {
                **key,
                "task_mode": task_mode,
                "pose_estimation_params": params,
                "pose_estimation_output_dir": output_dir,
            },
            skip_duplicates=skip_duplicates,
        )


@schema
class DLCPoseEstimation(dj.Computed):
    definition = """
    -> DLCPoseEstimationSelection
    ---
    pose_estimation_time: datetime  # time of generation of this set of DLC results
    """

    class BodyPartPosition(dj.Part):
        definition = """ # uses DeepLabCut h5 output for body part position
        -> DLCPoseEstimation
        -> DLCModel.BodyPart

        ---
        frame_index : longblob     # frame index in model
        x_pos       : longblob
        y_pos       : longblob
        z_pos=null  : longblob
        likelihood  : longblob
        """

    def make(self, key):
        """.populate() method will launch training for each PoseEstimationTask"""
        from .readers import dlc_reader

        # ID model and directories
        dlc_model = (Model & key).fetch1()

        task_mode, analyze_video_params, output_dir = (PoseEstimationTask & key).fetch1(
            "task_mode", "pose_estimation_params", "pose_estimation_output_dir"
        )
        analyze_video_params = analyze_video_params or {}
        output_dir = find_full_path(get_dlc_root_data_dir(), output_dir)
        video_filepaths = [
            find_full_path(get_dlc_root_data_dir(), fp).as_posix()
            for fp in (VideoRecording.File & key).fetch("file_path")
        ]
        project_path = find_full_path(
            get_dlc_root_data_dir(), dlc_model["project_path"]
        )

        # Triger PoseEstimation
        if task_mode == "trigger":
            dlc_reader.do_pose_estimation(
                video_filepaths,
                dlc_model,
                project_path,
                output_dir,
                **analyze_video_params,
            )
        dlc_result = dlc_reader.PoseEstimation(output_dir)
        creation_time = datetime.fromtimestamp(dlc_result.creation_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        body_parts = [
            {
                **key,
                "body_part": k,
                "frame_index": np.arange(dlc_result.nframes),
                "x_pos": v["x"],
                "y_pos": v["y"],
                "z_pos": v.get("z"),
                "likelihood": v["likelihood"],
            }
            for k, v in dlc_result.data.items()
        ]

        self.insert1({**key, "pose_estimation_time": creation_time})
        self.BodyPartPosition.insert(body_parts)

    @classmethod
    def get_trajectory(cls, key, body_parts="all"):
        """Returns a pandas dataframe of coordinates of the specified body_part(s)

        Parameters
        ----------
        key: A DataJoint query specifying one PoseEstimation entry. body_parts:
        Optional. Body parts as a list. If "all", all joints

        Returns
        -------
        df: multi index pandas dataframe with DLC scorer names, body_parts
            and x/y coordinates of each joint name for a camera_id, similar to output of
            DLC dataframe. If 2D, z is set of zeros
        """
        import pandas as pd

        model_name = key["model_name"]

        if body_parts == "all":
            body_parts = (cls.BodyPartPosition & key).fetch("body_part")
        else:
            body_parts = list(body_parts)

        df = None
        for body_part in body_parts:
            x_pos, y_pos, z_pos, likelihood = (
                cls.BodyPartPosition & {"body_part": body_part}
            ).fetch1("x_pos", "y_pos", "z_pos", "likelihood")
            if not z_pos:
                z_pos = np.zeros_like(x_pos)

            a = np.vstack((x_pos, y_pos, z_pos, likelihood))
            a = a.T
            pdindex = pd.MultiIndex.from_product(
                [[model_name], [body_part], ["x", "y", "z", "likelihood"]],
                names=["scorer", "bodyparts", "coords"],
            )
            frame = pd.DataFrame(a, columns=pdindex, index=range(0, a.shape[0]))
            df = pd.concat([df, frame], axis=1)
        return df

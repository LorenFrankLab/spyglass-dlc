from socket import if_indextoname
from urllib.parse import non_hierarchical
import numpy as np
import pandas as pd
import math
import datajoint as dj
from datetime import datetime
import deeplabcut
import pynwb
import os
import sys
import glob
import ruamel.yaml as yaml
from typing import List, Dict, OrderedDict
from pathlib import Path
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_behav import VideoFile
from spyglass.common.common_nwbfile import AnalysisNwbfile
from dgramling_dlc_project import BodyPart
from dgramling_dlc_model import DLCModel
from dlc_utils import find_full_path

schema = dj.schema('dgramling_dlc_pose_estimation')

@schema
class DLCPoseEstimationSelection(dj.Manual):
    definition = """
    -> VideoFile                           # Session -> Recording + File part table
    -> DLCModel                                    # Must specify a DLC project_path

    ---
    task_mode='load' : enum('load', 'trigger')  # load results or trigger computation
    video_path : varchar(120)                   # path to video file
    pose_estimation_output_dir='': varchar(255) # output dir relative to the root dir
    pose_estimation_params=null  : longblob     # analyze_videos params, if not default
    """

    # I think it makes more sense to just use a set output directory of 'cumulus/deeplabcut/pose_estimation/'
    # or a directory like that... or maybe on stelmo? depends on what Loren/Eric think 
    # @classmethod
    # def infer_output_dir(cls, key, relative=False, mkdir=False):
    #     """Return the expected pose_estimation_output_dir.

    #     With spaces in model name are replaced with hyphens.
    #     Based on convention: / video_dir / Device_{}_Recording_{}_Model_{}

    #     Parameters
    #     ----------
    #     key: DataJoint key specifying a pairing of VideoRecording and Model.
    #     relative (bool): Report directory relative to get_dlc_processed_data_dir().
    #     mkdir (bool): Default False. Make directory if it doesn't exist.
    #     """
    #     # TODO: add check to make sure interval_list_name refers to a single epoch
    #     # Or make key include epoch in and of itself instead of interval_list_name
    #     epoch = int(key['interval_list_name']
    #                 .replace('pos ', '')
    #                 .replace(' valid times', '')
    #                 ) + 1
    #     # Get video_filepath
    #     video_info = (VideoFile() &
    #                   {'nwb_file_name': key['nwb_file_name'],
    #                    'epoch': epoch}).fetch1()
    #     io = pynwb.NWBHDF5IO('/stelmo/nwb/raw/' +
    #                          video_info['nwb_file_name'], 'r')
    #     nwb_file = io.read()
    #     nwb_video = nwb_file.objects[video_info['video_file_object_id']]
    #     video_filepath = nwb_video.external_file[0]
    #     video_dir = os.path.dirname(video_filepath) + '/'
    #     video_filename = video_filepath.split(video_dir)[-1]
    #     recording_key = VideoFile & key
    #     # output to cumulus/deeplabcut/videos?
    #     output_dir = (
    #         '/cumulus/deeplabcut/videos'
    #         / (
    #             f'{video_filename}_model_'
    #             + key["model_name"].replace(" ", "-")
    #         )
    #     )
    #     if mkdir:
    #         output_dir.mkdir(parents=True, exist_ok=True)
    #     return output_dir.relative_to(processed_dir) if relative else output_dir
    
    @classmethod
    def get_video_path(cls, key):
        '''
        Given nwb_file_name and interval_list_name returns specified
        video file filename and path
        
        Parameters
        ----------
        key : dict
            Dictionary containing nwb_file_name and interval_list_name as keys
        Returns
        -------
        video_filepath : str
            path to the video file, including video filename
        video_filename : str
            filename of the video
        '''
        # TODO: add check to make sure interval_list_name refers to a single epoch
        # Or make key include epoch in and of itself instead of interval_list_name
        epoch = int(key['interval_list_name']
                    .replace('pos ', '')
                    .replace(' valid times', '')
                    ) + 1
        video_info = (VideoFile() &
                      {'nwb_file_name': key['nwb_file_name'],
                       'epoch': epoch}).fetch1()
        io = pynwb.NWBHDF5IO('/stelmo/nwb/raw/' +
                             video_info['nwb_file_name'], 'r')
        nwb_file = io.read()
        nwb_video = nwb_file.objects[video_info['video_file_object_id']]
        video_filepath = nwb_video.external_file[0]
        video_dir = os.path.dirname(video_filepath) + '/'
        video_filename = video_filepath.split(video_dir)[-1]
        return video_filepath, video_filename

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
        # Is output_dir even required if saving to set directory? Should there be a separate folder for each 
        # analyzed video?
        output_dir = cls.infer_output_dir(key, relative=relative, mkdir=mkdir)
        # TODO: figure out if a separate video_key is needed without portions of key that refer to model
        video_path, _ = cls.get_video_path(key)
        cls.insert1(
            {
                **key,
                "task_mode": task_mode,
                "pose_estimation_params": params,
                "video_path": video_path,
                "pose_estimation_output_dir": output_dir,
            },
            skip_duplicates=skip_duplicates,
        )


@schema
class DLCPoseEstimation(dj.Computed):
    definition = """
    -> DLCPoseEstimationSelection
    ---
    -> AnalysisNwbfile
    dlc_pose_estimation_object_id
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
        from dlc_reader import dlc_reader

        # ID model and directories
        dlc_model = (DLCModel & key).fetch1()
        key['analysis_file_name'] = AnalysisNwbfile().create(
            key['nwb_file_name'])
        task_mode, analyze_video_params, video_path, output_dir = (DLCPoseEstimationSelection & key).fetch1(
            "task_mode", "pose_estimation_params", "video_path","pose_estimation_output_dir"
        )
        analyze_video_params = analyze_video_params or {}
        output_dir = find_full_path(get_dlc_root_data_dir(), output_dir)

        project_path = find_full_path(
            get_dlc_root_data_dir(), dlc_model["project_path"]
        )

        # Triger PoseEstimation
        if task_mode == "trigger":
            dlc_reader.do_pose_estimation(
                video_path,
                dlc_model,
                project_path,
                output_dir,
                **analyze_video_params,
            )
        dlc_result = dlc_reader.PoseEstimation(output_dir)
        creation_time = datetime.fromtimestamp(dlc_result.creation_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        # body_parts = [
        #     {
        #         **key,
        #         "body_part": k,
        #         "frame_index": np.arange(dlc_result.nframes),
        #         "x_pos": v["x"],
        #         "y_pos": v["y"],
        #         "z_pos": v.get("z"),
        #         "likelihood": v["likelihood"],
        #     }
        #     for k, v in dlc_result.data.items()
        # ]
        # TODO: determine where to add timestamps to the dataframe. 
        nwb_analysis_file = AnalysisNwbfile()
        key['dlc_pose_estimation_object_id'] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=dlc_result.df,
        )

        nwb_analysis_file.add(
            nwb_file_name=key['nwb_file_name'],
            analysis_file_name=key['analysis_file_name'])

        #  Do I need this
        # AnalysisNwbfile().add(
        #     key['nwb_file_name'], key['analysis_file_name'])

        self.insert1(key)
        self.insert1({**key, "pose_estimation_time": creation_time})
        # self.BodyPartPosition.insert(body_parts)
    
    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'),
                         *attrs, **kwargs)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]['linearized_position'].set_index('time')

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

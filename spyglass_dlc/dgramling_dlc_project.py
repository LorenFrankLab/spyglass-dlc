import numpy as np
import datajoint as dj
import os
import glob
import ruamel.yaml
from itertools import combinations
from typing import List
from pathlib import Path
from spyglass.common.common_lab import LabTeam
from .dlc_utils import _convert_mp4


schema = dj.schema("dgramling_dlc_project")


@schema
class BodyPart(dj.Manual):
    """Holds bodyparts for use in DeepLabCut models"""

    definition = """
    bodypart                : varchar(32)
    ---
    bodypart_description='' : varchar(80)
    """
    # TODO: add option to insert from pre-exisiting model


@schema
class DLCProject(dj.Manual):
    """Table to facilitate creation of a new DeepLabCut model.
    With ability to edit config, extract frames, label frames
    """

    definition = """
    project_name     : varchar(100) # name of DLC project
    ---
    -> LabTeam
    bodyparts        : blob         # list of bodyparts to label
    frames_per_video : int          # number of frames to extract from each video
    config_path      : varchar(120) # path to config.yaml for model
    """

    class BodyPart(dj.Part):
        """Part table to hold bodyparts used in each project."""

        definition = """
        -> DLCProject
        -> BodyPart
        """

    class File(dj.Part):
        definition = """
        # Paths of training files (e.g., labeled pngs, CSV or video)
        -> DLCProject
        file_name: varchar(40) # Concise name to describe file
        file_ext : enum("mp4", "csv", "h5") # extension of file
        ---
        file_path: varchar(255)
        """

    def insert1(self, key, **kwargs):
        assert isinstance(key["project_name"], str), "project_name must be a string"
        project_names_in_use = np.unique(self.fetch("project_name"))
        assert (
            key["project_name"] not in project_names_in_use
        ), f"project name: {key['project_name']} is already in use."
        if not bool(LabTeam() & {"team_name": key["team_name"]}):
            raise ValueError(f"team_name: {key['team_name']} does not exist in LabTeam")
        assert isinstance(
            key["frames_per_video"], int
        ), "frames_per_video must be of type `int`"
        super().insert1(key, **kwargs)

    @classmethod
    def insert_existing_project(
        cls,
        project_name: str,
        bodyparts: List,
        lab_team: str,
        config_path: str,
        frames_per_video: int = None,
        add_to_files: bool = True,
        **kwargs,
    ):
        """
        insert an existing project into DLCProject table.
        Parameters
        ----------
        project_name : str
            user-friendly name of project
        bodyparts : list
            list of bodyparts to label. Should match bodyparts in BodyPart table
        lab_team : str
            name of lab team. Should match an entry in LabTeam table
        project_path : str
            path to project directory
        """
        # Read config
        from deeplabcut.utils.auxiliaryfunctions import read_config

        cfg = read_config(config_path)
        bodyparts_to_add = [
            bodypart for bodypart in bodyparts if bodypart not in cfg["bodyparts"]
        ]
        all_bodyparts = bodyparts_to_add + cfg["bodyparts"]
        for bodypart in all_bodyparts:
            if not bool(BodyPart() & {"bodypart": bodypart}):
                raise ValueError(f"bodypart: {bodypart} not found in BodyPart table")
        # check bodyparts are in config, if not add
        if len(bodyparts_to_add) > 0:
            add_to_config(config_path, bodyparts=bodyparts_to_add)
        # Get frames per video from config. If passed as arg, check match
        if frames_per_video:
            if frames_per_video != cfg["numframes2pick"]:
                add_to_config(config_path, **{"numframes2pick": frames_per_video})

        key = {
            "project_name": project_name,
            "team_name": lab_team,
            "bodyparts": bodyparts,
            "config_path": config_path,
            "frames_per_video": frames_per_video,
        }
        cls.insert1(key, skip_duplicates=True)
        cls.BodyPart.insert((project_name, bp) for bp in all_bodyparts)
        if add_to_files:
            del key["bodyparts"]
            del key["team_name"]
            del key["config_path"]
            del key["frames_per_video"]
            # Check for training files to add
            cls.add_training_files(key)

    @classmethod
    def insert_new_project(
        cls,
        project_name: str,
        bodyparts: List,
        lab_team: str,
        frames_per_video: int,
        video_path: str,
        project_directory: str = "cumulus/deeplabcut/",
        convert_video=False,
        video_names: List = None,
        **kwargs,
    ):
        """
        insert a new project into DLCProject table.
        Parameters
        ----------
        project_name : str
            user-friendly name of project
        bodyparts : list
            list of bodyparts to label. Should match bodyparts in BodyPart table
        lab_team : str
            name of lab team. Should match an entry in LabTeam table
        project_directory : str
            directory where to create project.
            (Default is '/cumulus/deeplabcut/')
        frames_per_video : int
            number of frames to extract from each video
        video_path : str
            directory where videos to create model from are stored
        convert_video : bool
            if True will convert videos in video_path to .MP4 format from .h264
            (Default is False)
        video_names : list
            (Default is None) list of video names to extract frames from
            If not None, will limit videos within video_path to use
        """
        skeleton_node = None
        # TODO: this needs to be cleaned up
        if convert_video:
            from .dlc_utils import check_videofile

            check_videofile(video_path=video_path, output_path=video_path)
        videos = glob.glob(video_path + "*.mp4")
        if len(videos) < 1:
            raise ValueError(f"no .mp4 videos found in{video_path}")
        if video_names:
            videos = [
                video for video in videos if any(map(video.__contains__, video_names))
            ]
        from deeplabcut import create_new_project

        config_path = create_new_project(
            project_name,
            lab_team,
            videos,
            working_directory=project_directory,
            copy_videos=True,
            multianimal=False,
        )
        for bodypart in bodyparts:
            if not bool(BodyPart() & {"bodypart": bodypart}):
                raise ValueError(f"bodypart: {bodypart} not found in BodyPart table")
        config_kwargs = {"numframes2pick": frames_per_video, "dotsize": 3}.update(
            kwargs
        )
        add_to_config(
            config_path, bodyparts, skeleton_node=skeleton_node, **config_kwargs
        )
        key = {
            "project_name": project_name,
            "team_name": lab_team,
            "bodyparts": bodyparts,
            "config_path": config_path,
            "numframes2pick": frames_per_video,
        }
        if not os.path.exists(key["video_path"]):
            raise OSError(f"{key['video_path']} does not exist")
        cls.insert1(key)
        cls.BodyPart.insert((project_name, bp) for bp in bodyparts)

    @classmethod
    def add_training_files(cls, key):
        """Add training videos and labeled frames .h5 and .csv to DLCProject.File"""
        config_path = (cls & {"project_name": key["project_name"]}).fetch1(
            "config_path"
        )
        from deeplabcut.utils.auxiliaryfunctions import read_config

        cfg = read_config(config_path)
        video_names = list(cfg["video_sets"].keys())
        training_files = []
        for video in video_names:
            video_name = os.path.splitext(
                video.split(os.path.dirname(video) + "/")[-1]
            )[0]
            training_files.extend(
                glob.glob(
                    f"{cfg['project_path']}/labeled-data/{video_name}/*Collected*"
                )
            )
        for video in video_names:
            key["file_name"] = f'video_{os.path.splitext(video.split("/")[-1])[0]}'
            key["file_ext"] = os.path.splitext(video.split("/")[-1])[-1].split(".")[-1]
            key["file_path"] = video
            cls.File.insert1(key, skip_duplicates=True)
        if len(training_files) > 0:
            for file in training_files:
                video_name = os.path.dirname(file).split("/")[-1]
                file_type = os.path.splitext(
                    file.split(os.path.dirname(file) + "/")[-1]
                )[-1].split(".")[-1]
                key["file_name"] = f"{video_name}_labeled_data"
                key["file_ext"] = file_type
                key["file_path"] = file
                cls.File.insert1(key, skip_duplicates=True)
        else:
            Warning("No training files to add")

    def run_extract_frames(self, key, **kwargs):
        """Convenience function to launch DLC GUI for extracting frames.
        Must be run on local machine to access GUI,
        cannot be run through ssh tunnel
        """
        config_path = (self & key).fetch1("config_path")
        from deeplabcut import extract_frames

        extract_frames(config_path, **kwargs)

    def run_label_frames(self, key):
        """Convenience function to launch DLC GUI for labeling frames.
        Must be run on local machine to access GUI,
        cannot be run through ssh tunnel
        """
        config_path = (self & key).fetch1("config_path")
        from deeplabcut import label_frames

        label_frames(config_path)

    def check_labels(self, key):
        """Convenience function to check labels on
        previously extracted and labeled frames
        """
        config_path = (self & key).fetch1("config_path")
        from deeplabcut import check_labels

        check_labels(config_path)


def add_to_config(config, bodyparts: List = None, skeleton_node: str = None, **kwargs):
    """
    Add necessary items to the config.yaml for the model
    Parameters
    ----------
    config : str
        Path to config.yaml
    bodyparts : list
        list of bodyparts to add to model
    skeleton_node : str
        (default is None) node to link LEDs in skeleton
    kwargs : dict
        Other parameters of config to modify in key:value pairs
    """

    yaml = ruamel.yaml.YAML()
    with open(config) as fp:
        data = yaml.load(fp)
    if bodyparts:
        data["bodyparts"] = bodyparts
        led_parts = [element for element in bodyparts if "LED" in element]
        if skeleton_node is not None:
            bodypart_skeleton = [
                list(link)
                for link in combinations(led_parts, 2)
                if skeleton_node in link
            ]
        else:
            bodypart_skeleton = list(combinations(led_parts, 2))
        other_parts = list(set(bodyparts) - set(led_parts))
        bodypart_skeleton.append(other_parts)
        data["skeleton"] = bodypart_skeleton
    for kwarg, val in kwargs.items():
        if not isinstance(kwarg, str):
            kwarg = str(kwarg)
        data[kwarg] = val
    with open(config, "w") as fw:
        yaml.dump(data, fw)

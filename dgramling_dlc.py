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
import ruamel.yaml
from itertools import combinations
from typing import List, Dict, OrderedDict
from pathlib import Path
import spyglass.common as sgc
from spyglass.common.common_ephys import Raw
from spyglass.common.common_lab import LabTeam
from spyglass.common.common_interval import IntervalList, interval_list_contains
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session
from spyglass.common.common_task import TaskEpoch
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.nwb_helper_fn import get_all_spatial_series, get_data_interface, get_nwb_file

schema = dj.schema('dgramling_dlc')

@schema
class BodyPart(dj.Manual):
    """Holds bodyparts for use in DeepLabCut models"""
    definition="""
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
    definition="""
    project_name     : varchar(100) # name of DLC project
    ---
    -> LabTeam
    bodyparts        : blob         # list of bodyparts to label
    frames_per_video : int          # number of frames to extract from each video
    config_path      : varchar(120) # path to config.yaml for model
    """
    # TODO: add option to load project that has already been created outside of datajoint
    class BodyPart(dj.Part):
        """Part table to hold bodyparts used in each project.
        """
        definition="""
        -> DLCProject
        -> BodyPart
        """
    
    @classmethod
    def insert_new_project(cls, project_name: str, bodyparts: List,
                            lab_team: str, frames_per_video: int,
                            video_path: str,
                            project_directory: str='cumulus/deeplabcut/',
                            convert_video=False, video_names: List=None,
                            **kwargs):
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
        assert isinstance(project_name, str), 'project_name must be a string'
        if not bool(LabTeam() & {'team_name' : lab_team}):
            raise ValueError(f'lab_team: {lab_team} does not exist in LabTeam')
        assert isinstance(frames_per_video, int), 'frames_per_video must be of type `int`'
        if not os.path.exists(video_path):
            raise OSError(f'{video_path} does not exist')
        # VideoSet.add_videos needs to be fixed, but this should perform conversion, etc..
        # VideoSet.add_videos(video_names, filenames=)

        if convert_video:
            raise NotImplementedError(f'argument `convert_video` has yet to be tested')
            videos_to_convert = glob.glob(video_path+ '*.h264')
            (convert_mp4(video, video_path, video_path, 'mp4') for video in videos_to_convert)
        videos = glob.glob(video_path+'*.mp4')
        if len(videos) < 1:
            raise ValueError(f'no .mp4 videos found in{video_path}')
        if video_names:
            videos = [video for video in videos if any(map(video.__contains__, video_names))]
        
        config_path = deeplabcut.create_new_project(project_name, lab_team,
                                                    videos,
                                                    working_directory=project_directory,
                                                    copy_videos=True, multianimal=False)
        for bodypart in bodyparts:
            if not bool(BodyPart() & {'bodypart' : bodypart}):
                raise ValueError(f'bodypart: {bodypart} not found in BodyPart table')
        key = {'project_name': project_name, 'lab_team': lab_team,
            'bodyparts': bodyparts, 'config_path': config_path, 
            'frames_per_video' : frames_per_video}
        cls.insert1(key)
        cls.BodyPart.insert((project_name, bp) for bp in bodyparts)
        add_to_config(config_path, bodyparts, skeleton_node=skeleton_node,
                    numframes2pick=frames_per_video, dotsize=3)
    
    def run_extract_frames(self, key, **kwargs):
        config_path = (self & key).fetch1('config_path')
        deeplabcut.extract_frames(config_path, **kwargs)
    
    def run_label_frames(self, key):
        config_path = (self & key).fetch1('config_path')
        deeplabcut.label_frames(config_path)
    
    def check_labels(self, key):
        config_path = (self & key).fetch1('config_path')
        deeplabcut.check_labels(config_path)
    

def add_to_config(config, bodyparts, skeleton_node: str=None, **kwargs):
    '''
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
    '''

    yaml = ruamel.yaml.YAML()
    with open(config) as fp:
        data = yaml.load(fp)
    data['bodyparts'] = bodyparts
    led_parts = [element for element in bodyparts if 'LED' in element]
    if skeleton_node is not None:
        bodypart_skeleton = [list(link) for link in combinations(led_parts,2) 
                             if skeleton_node in link]
    else:
        bodypart_skeleton = list(combinations(led_parts,2))
    other_parts = list(set(bodyparts) - set(led_parts))
    bodypart_skeleton.append(other_parts)
    data['skeleton'] = bodypart_skeleton
    for kwarg, val in kwargs.items():
        if not isinstance(kwarg, str):
            kwarg = str(kwarg)
        data[kwarg] = val
    with open(config, 'w') as fw:
        yaml.dump(data, fw)

def convert_mp4(filename: str, video_path: str, dest_path: str, 
                videotype: str, count_frames=False, return_output=True):
    '''converts video to mp4 using passthrough for frames
    Parameters
    ----------
    filename: str
        filename of video including suffix (e.g. .h264)
    video_path: str
        path of input video excluding filename
    dest_path: str
        path of output video excluding filename, but no suffix (e.g. no '.mp4')
    videotype: str
        str of filetype to convert to (currently only accepts 'mp4')
    count_frames: bool
        if True counts the number of frames in both videos instead of packets
    return_output: bool
        if True returns the destination filename
    '''

    import subprocess
    orig_filename = filename
    video_path = Path(video_path + filename)
    if videotype not in ['mp4']:
        raise NotImplementedError
    dest_filename = os.path.splitext(filename)[0]
    if '.1' in dest_filename:
        dest_filename = os.path.splitext(dest_filename)[0]
    dest_path = Path(dest_path + dest_filename + '.' + videotype)
    convert_command = f"ffmpeg -vsync passthrough -i {video_path} -codec copy {dest_path}"
    os.system(convert_command)
    print(f'finished converting {filename}')
    print(f'Checking that number of packets match between {orig_filename} and {dest_filename}')
    num_packets = []
    for ind, file in enumerate([video_path, dest_path]):
        packets_command = ['ffprobe', '-v', 'error', '-select_streams',
                        'v:0', '-count_packets', '-show_entries',
                        'stream=nb_read_packets', '-of', 'csv=p=0', file]
        frames_command = ['ffprobe', '-v', 'error', '-select_streams',
                         'v:0', '-count_frames', '-show_entries',
                         'stream=nb_read_frames', '-of', 'csv=p=0', file]
        if count_frames:
            p = subprocess.Popen(frames_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            p = subprocess.Popen(packets_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        num_packets.append(int(out.decode("utf-8").split("\n")[0]))
    print(f'Number of packets in {orig_filename}: {num_packets[0]}, {dest_filename}: {num_packets[1]}')
    assert num_packets[0] == num_packets[1]
    if return_output:
        return dest_filename


@schema
class VideoSet(dj.Manual):
    definition = """
    # Table to hold videos used to train DeepLabCut Models
    # Theoretically allows for adding of new materials for training...
    video_name: varchar(40)  # Concise description of video
    ---
    file_path: varchar(255)  # Path to the video file
    environment: varchar(40) # Environment where video was recorded..should pull from some table
    """
    
    #TODO: maybe add seconday key that is path to extracted frames
    @classmethod
    def add_videos(cls, video_names: List, filenames: List, path: str, environment: str):
        ''' Enforce .MP4 requirement for all videos
        This is broken until I can map video_names to filenames
        '''
        added_files = []
        all_files = glob.glob(path + '*')
        video_files = [file for file in all_files if any(map(file.__contains__, filenames))]
        if len(video_files) < 1:
            raise Exception('No video file found')
        if len(video_files) > 1:
            assert len(video_names) == len(video_files), 'need to provide a video name for each file'
            print(f'found {len(video_files)} matching files to add')
            # TODO figure out how to map video_names to video_files
            for file in video_files:
                base_filename, file_ext = os.path.splitext(file)
                if file_ext.lower() != '.mp4':
                    filename = convert_mp4(file, path, path, 'mp4')
                else:
                    filename = base_filename
                # Check that path is valid
                if os.path.exists(Path(path + filename)):
                    cls.insert1({'video_name' : video_name, 'file_path': Path(path + filename),
                                 'environment': environment})
                    added_files.append(path + filename)
                else:
                    # TODO: move this to a more appropriate location within the function
                    raise Warning(f'Tried to insert {filename}, but the path was not valid')
        return added_files


class DLCModelTraining(dj.Manual):
    """Table to facilitate creation of a new DeepLabCut model.
    With ability to edit config, extract frames, label frames
    """
    definition="""
    project_name     : varchar(100) # name of DLC project
    ---
    -> LabTeam
    bodyparts        : blob         # list of bodyparts to label
    frames_per_video : int          # number of frames to extract from each video
    config_path      : varchar(120) # path to config.yaml for model
    """

@schema
class DLCModelTrainingParams(dj.Lookup):
    definition = """
    # Parameters to specify a DLC model training instance
    # For DLC ≤ v2.0, include scorer_lecacy = True in params
    dlc_training_params_name      : varchar(50) # descriptive name of parameter set
    ---
    params                        : longblob    # dictionary of all applicable parameters
    """

    required_parameters = ("shuffle", "trainingsetindex")
    skipped_parameters = ("project_path", "video_sets")

    @classmethod
    def insert_new_params(
        cls, paramset_name: str, params: dict, paramset_idx: int = None
    ):
        """
        Insert a new set of training parameters into dlc.TrainingParamSet.

        Parameters
        ----------
        paramset_name (str): Description of parameter set to be inserted
        params (dict): Dictionary including all settings to specify model training.
                       Must include shuffle & trainingsetindex b/c not in config.yaml.
                       project_path and video_sets will be overwritten by config.yaml.
                       Note that trainingsetindex is 0-indexed
        """

        for required_param in cls.required_parameters:
            assert required_param in params, (
                "Missing required parameter: " + required_param
            )
        for skipped_param in cls.skipped_parameters:
            if skipped_param in params:
                params.pop(skipped_param)

        if paramset_idx is None:
            paramset_idx = (
                dj.U().aggr(cls, n="max(paramset_idx)").fetch1("n") or 0
            ) + 1

        param_dict = {
            "dlc_training_params_name": paramset_name,
            "params": params,
        }
        param_query = cls & {
            "dlc_training_params_name": param_dict["dlc_training_params_name"]}
        # If the specified param-set already exists
        # Not sure we need this part, as much just a check if the name is the same
        if param_query:
            existing_paramset_name = param_query.fetch1("dlc_training_params_name")
            if existing_paramset_name == paramset_name:  # If existing_idx same:
                return print(
                    f'New param set not added\n'
                    f'A param set with name: {paramset_name} already exists')
        else:
            cls.insert1(param_dict)  # if duplicate, will raise duplicate error
            # if this will raise duplicate error, why is above check needed? @datajoint


@schema
class DLCModelTrainingSelection(dj.Manual):
    definition = """      # Specification for a DLC model training instance
    -> VideoSet           # labeled video(s) for training
    -> DLCModelTrainingParams
    training_id     : int
    ---
    model_prefix='' : varchar(32)
    project_path='' : varchar(255) # DLC's project_path in config relative to root
    """


@schema
class DLCModelTraining(dj.Computed):
    definition = """
    -> DLCModelTrainingSelection
    ---
    latest_snapshot: int unsigned # latest exact snapshot index (i.e., never -1)
    config_template: longblob     # stored full config file
    """

    # To continue from previous training snapshot, devs suggest editing pose_cfg.yml
    # https://github.com/DeepLabCut/DeepLabCut/issues/70

    def make(self, key):
        """Launch training for each train.TrainingTask training_id via `.populate()`."""
        project_path, model_prefix = (DLCModelTrainingSelection & key).fetch1(
            "project_path", "model_prefix"
        )
        from deeplabcut import train_network
        from .readers import dlc_reader

        try:
            from deeplabcut.utils.auxiliaryfunctions import get_model_folder
        except ImportError:
            from deeplabcut.utils.auxiliaryfunctions import (
                GetModelFolder as get_model_folder,
            )
        project_path = 
        project_path = find_full_path(get_dlc_root_data_dir(), project_path)

        # ---- Build and save DLC configuration (yaml) file ----
        _, dlc_config = dlc_reader.read_yaml(project_path)  # load existing
        dlc_config.update((DLCModelTrainingParams & key).fetch1("params"))
        dlc_config.update(
            {
                "project_path": project_path.as_posix(),
                "modelprefix": model_prefix,
                "train_fraction": dlc_config["TrainingFraction"][
                    int(dlc_config["trainingsetindex"])
                ],
                "training_filelist_datajoint": [  # don't overwrite origin video_sets
                    find_full_path(get_dlc_root_data_dir(), fp).as_posix()
                    for fp in (VideoSet & key).fetch("file_path")
                ],
            }
        )
        # Write dlc config file to base project folder
        dlc_cfg_filepath = dlc_reader.save_yaml(project_path, dlc_config)

        # ---- Trigger DLC model training job ----
        train_network_input_args = list(inspect.signature(train_network).parameters)
        train_network_kwargs = {
            k: v for k, v in dlc_config.items() if k in train_network_input_args
        }
        for k in ["shuffle", "trainingsetindex", "maxiters"]:
            train_network_kwargs[k] = int(train_network_kwargs[k])

        try:
            train_network(dlc_cfg_filepath, **train_network_kwargs)
        except KeyboardInterrupt:  # Instructions indicate to train until interrupt
            print("DLC training stopped via Keyboard Interrupt")

        snapshots = list(
            (
                project_path
                / get_model_folder(
                    trainFraction=dlc_config["train_fraction"],
                    shuffle=dlc_config["shuffle"],
                    cfg=dlc_config,
                    modelprefix=dlc_config["modelprefix"],
                )
                / "train"
            ).glob("*index*")
        )
        max_modified_time = 0
        # DLC goes by snapshot magnitude when judging 'latest' for evaluation
        # Here, we mean most recently generated
        for snapshot in snapshots:
            modified_time = os.path.getmtime(snapshot)
            if modified_time > max_modified_time:
                latest_snapshot = int(snapshot.stem[9:])
                max_modified_time = modified_time

        self.insert1(
            {**key, "latest_snapshot": latest_snapshot, "config_template": dlc_config}
        )








"""----From Datajoint-Elements----"""
@schema
class Model(dj.Manual):
    definition = """
    model_name           : varchar(64)  # User-friendly model name
    -> ModelTraining
    ---
    task                 : varchar(32)  # Task in the config yaml
    date                 : varchar(16)  # Date in the config yaml
    iteration            : int          # Iteration/version of this model
    snapshotindex        : int          # which snapshot for prediction (if -1, latest)
    shuffle              : int          # Shuffle (1) or not (0)
    trainingsetindex     : int          # Index of training fraction list in config.yaml
    unique index (task, date, iteration, shuffle, snapshotindex, trainingsetindex)
    scorer               : varchar(64)  # Scorer/network name - DLC's GetScorerName()
    config_template      : longblob     # Dictionary of the config for analyze_videos()
    project_path         : varchar(255) # DLC's project_path in config relative to root
    model_prefix=''      : varchar(32)
    model_description='' : varchar(1000)
    """
    # project_path is the only item required downstream in the pose schema

    class BodyPart(dj.Part):
        definition = """
        -> master
        -> BodyPart
        """

    @classmethod
    def insert_new_model(
        cls,
        model_name: str,
        dlc_config,
        *,
        shuffle: int,
        trainingsetindex,
        project_path=None,
        model_description="",
        model_prefix="",
        paramset_idx: int = None,
        prompt=True,
        params=None,
    ):
        """Insert new model into the dlc.Model table.

        Parameters
        ----------
        model_name (str): User-friendly name for this model.
        dlc_config (str or dict):  path to a config.y*ml, or dict of such contents.
        shuffle (int): Shuffled or not as 1 or 0.
        trainingsetindex (int): Index of training fraction list in config.yaml.
        model_description (str): Optional. Description of this model.
        model_prefix (str): Optional. Filename prefix used across DLC project
        body_part_descriptions (list): Optional. List of descriptions for BodyParts.
        paramset_idx (int): Optional. Index from the TrainingParamSet table
        prompt (bool): Optional.
        params (dict): Optional. If dlc_config is path, dict of override items
        """
        from deeplabcut.utils.auxiliaryfunctions import GetScorerName
        from .readers import dlc_reader

        # handle dlc_config being a yaml file
        if not isinstance(dlc_config, dict):
            dlc_config_fp = find_full_path(get_dlc_root_data_dir(), Path(dlc_config))
            assert dlc_config_fp.exists(), (
                "dlc_config is neither dict nor filepath" + f"\n Check: {dlc_config_fp}"
            )
            if dlc_config_fp.suffix in (".yml", ".yaml"):
                with open(dlc_config_fp, "rb") as f:
                    dlc_config = yaml.safe_load(f)
            if isinstance(params, dict):
                dlc_config.update(params)

        # ---- Get and resolve project path ----
        project_path = find_full_path(
            get_dlc_root_data_dir(), dlc_config.get("project_path", project_path)

        )
        dlc_config["project_path"] = str(project_path)  # update if different
        root_dir = find_root_directory(get_dlc_root_data_dir(), project_path)

        # ---- Verify config ----
        needed_attributes = [
            "Task",
            "date",
            "iteration",
            "snapshotindex",
            "TrainingFraction",
        ]
        for attribute in needed_attributes:
            assert attribute in dlc_config, f"Couldn't find {attribute} in config"

        # ---- Get scorer name ----
        # "or 'f'" below covers case where config returns None. str_to_bool handles else
        scorer_legacy = str_to_bool(dlc_config.get("scorer_legacy", "f"))


        dlc_scorer = GetScorerName(
            cfg=dlc_config,
            shuffle=shuffle,
            trainFraction=dlc_config["TrainingFraction"][int(trainingsetindex)],
            modelprefix=model_prefix,
        )[scorer_legacy]
        if dlc_config["snapshotindex"] == -1:
            dlc_scorer = "".join(dlc_scorer.split("_")[:-1])

        # ---- Insert ----
        model_dict = {
            "model_name": model_name,
            "model_description": model_description,
            "scorer": dlc_scorer,
            "task": dlc_config["Task"],
            "date": dlc_config["date"],
            "iteration": dlc_config["iteration"],
            "snapshotindex": dlc_config["snapshotindex"],
            "shuffle": shuffle,
            "trainingsetindex": int(trainingsetindex),
            "project_path": project_path.relative_to(root_dir).as_posix(),
            "paramset_idx": paramset_idx,
            "config_template": dlc_config,
        }

        # -- prompt for confirmation --
        if prompt:
            print("--- DLC Model specification to be inserted ---")
            for k, v in model_dict.items():
                if k != "config_template":
                    print("\t{}: {}".format(k, v))
                else:
                    print("\t-- Template/Contents of config.yaml --")
                    for k, v in model_dict["config_template"].items():
                        print("\t\t{}: {}".format(k, v))

        if (
            prompt
            and dj.utils.user_choice("Proceed with new DLC model insert?") != "yes"
        ):
            print("Canceled insert.")
            return
        # ---- Save DJ-managed config ----
        _ = dlc_reader.save_yaml(project_path, dlc_config)
        # ____ Insert into table ----
        with cls.connection.transaction:
            cls.insert1(model_dict)
            # Returns array, so check size for unambiguous truth value
            if BodyPart.extract_new_body_parts(dlc_config, verbose=False).size > 0:
                BodyPart.insert_from_config(dlc_config, prompt=prompt)
            cls.BodyPart.insert((model_name, bp) for bp in dlc_config["bodyparts"])

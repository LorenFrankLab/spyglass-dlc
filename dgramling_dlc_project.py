from socket import if_indextoname
from urllib.parse import non_hierarchical
import numpy as np
import pandas as pd
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
from spyglass.common.common_lab import LabTeam


schema = dj.schema('dgramling_dlc_project')

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
class TrainingSet(dj.Computed):
    definition = """
    -> DLCProject
    training_set_name: varchar(40) # Concise name for training materials
    """

    class File(dj.Part):
        definition = """
        # Paths of training files (e.g., labeled pngs, CSV or video)
        -> TrainingSet
        file_name: varchar(40) # Concise name to describe file
        ---
        file_path: varchar(255)
        """
    def make(self, key):
        self.insert1(key)
        config_path = (DLCProject() & key).fetch1('config_path')
        from deeplabcut.utils.auxiliaryfunctions import read_config
        cfg = read_config(config_path)
        video_names = list(cfg['video_sets'].keys())
        training_files = []
        for video in video_names:
            video_name = os.path.splitext(video.split(os.path.dirname(video) + '/')[-1])[0]
            training_files.append(glob.glob(f"{cfg['project_path']}/labeled-data/{video_name}/*Collected*"))
        for ind, video in enumerate(video_names):
            key['file_name'] = f'video{ind+1}'
            key['file_path'] = video
            self.File.insert1(key)
        for file in training_files:
            file_type = os.path.splitext(file.split(os.path.dirname(file) + '/')[-1])[-1]
            key['file_name'] = f'labeled_data_{file_type}'
            key['file_path'] = file
            self.File.insert1(key)

# @schema
# class VideoSet(dj.Manual):
#     definition = """
#     # Table to hold videos used to train DeepLabCut Models
#     # Theoretically allows for adding of new materials for training...
#     video_name: varchar(40)  # Concise description of video
#     ---
#     file_path: varchar(255)  # Path to the video file
#     environment: varchar(40) # Environment where video was recorded..should pull from some table
#     """
    
#     #TODO: maybe add seconday key that is path to extracted frames
#     @classmethod
#     def add_videos(cls, video_names: List, filenames: List, path: str, environment: str):
#         ''' Enforce .MP4 requirement for all videos
#         This is broken until I can map video_names to filenames
#         '''
#         added_files = []
#         all_files = glob.glob(path + '*')
#         video_files = [file for file in all_files if any(map(file.__contains__, filenames))]
#         if len(video_files) < 1:
#             raise Exception('No video file found')
#         if len(video_files) > 1:
#             assert len(video_names) == len(video_files), 'need to provide a video name for each file'
#             print(f'found {len(video_files)} matching files to add')
#             # TODO figure out how to map video_names to video_files
#             for file in video_files:
#                 base_filename, file_ext = os.path.splitext(file)
#                 if file_ext.lower() != '.mp4':
#                     filename = convert_mp4(file, path, path, 'mp4')
#                 else:
#                     filename = base_filename
#                 # Check that path is valid
#                 if os.path.exists(Path(path + filename)):
#                     cls.insert1({'video_name' : video_name, 'file_path': Path(path + filename),
#                                  'environment': environment})
#                     added_files.append(path + filename)
#                 else:
#                     # TODO: move this to a more appropriate location within the function
#                     raise Warning(f'Tried to insert {filename}, but the path was not valid')
#         return added_files
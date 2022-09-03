import numpy as np
import pandas as pd
import datajoint as dj
import inspect
import os
import ruamel.yaml as yaml
from itertools import combinations
from pathlib import Path
from .dlc_utils import find_full_path
from .dgramling_dlc_project import DLCProject

schema = dj.schema("dgramling_dlc_training")


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
    def insert_new_params(cls, paramset_name: str, params: dict):
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

        param_dict = {
            "dlc_training_params_name": paramset_name,
            "params": params,
        }
        param_query = cls & {
            "dlc_training_params_name": param_dict["dlc_training_params_name"]
        }
        # If the specified param-set already exists
        # Not sure we need this part, as much just a check if the name is the same
        if param_query:
            existing_paramset_name = param_query.fetch1("dlc_training_params_name")
            if existing_paramset_name == paramset_name:  # If existing name same:
                return print(
                    f"New param set not added\n"
                    f"A param set with name: {paramset_name} already exists"
                )
        else:
            cls.insert1(param_dict)  # if duplicate, will raise duplicate error
            # if this will raise duplicate error, why is above check needed? @datajoint


@schema
class DLCModelTrainingSelection(dj.Manual):
    definition = """      # Specification for a DLC model training instance
    -> DLCProject
    -> DLCModelTrainingParams
    training_id     : int # unique integer, 
    # allows for multiple training runs for a specific parameter set and project
    ---
    model_prefix='' : varchar(32)
    project_path='' : varchar(255) # DLC's project_path in config relative to root
    """
    # What is the purpose of the training_id and project_path


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
        # Not sure what either of these accomplish
        project_path, model_prefix = (DLCModelTrainingSelection & key).fetch1(
            "project_path", "model_prefix"
        )
        from deeplabcut import train_network
        import dlc_reader
        from deeplabcut.utils.auxiliaryfunctions import read_config

        try:
            from deeplabcut.utils.auxiliaryfunctions import get_model_folder
        except ImportError:
            from deeplabcut.utils.auxiliaryfunctions import (
                GetModelFolder as get_model_folder,
            )
        config_path = (DLCProject() & key).fetch1("config_path")
        dlc_config = read_config(config_path)
        project_path = dlc_config["project_path"]

        # ---- Build and save DLC configuration (yaml) file ----
        _, dlc_config = dlc_reader.read_yaml(project_path)
        dlc_config.update((DLCModelTrainingParams & key).fetch1("params"))
        dlc_config.update(
            {
                "project_path": Path(project_path),
                "modelprefix": model_prefix,
                "train_fraction": dlc_config["TrainingFraction"][
                    int(dlc_config["trainingsetindex"])
                ],
                "training_filelist_datajoint": [  # don't overwrite origin video_sets
                    Path(fp) for fp in (DLCProject.File & key).fetch("file_path")
                ],
            }
        )
        # Write dlc config file to base project folder
        # TODO: need to make sure this will work
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

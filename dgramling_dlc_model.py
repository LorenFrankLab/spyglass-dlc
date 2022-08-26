from socket import if_indextoname
from urllib.parse import non_hierarchical
import numpy as np
import pandas as pd
import math
import datajoint as dj
from datajoint.errors import DataJointError
import deeplabcut
import pynwb
import os
import sys
import glob
import ruamel.yaml as yaml
from typing import List, Dict, OrderedDict
from pathlib import Path
from dlc_decorators import accepts
from dlc_utils import (
    find_full_path,
    get_dlc_processed_data_dir,
    get_dlc_root_data_dir,
    find_root_directory,
)
from spyglass.common.common_lab import LabTeam
from dgramling_dlc_project import BodyPart
from dgramling_dlc_training import DLCModelTraining

schema = dj.schema("dgramling_dlc_model")


@schema
class DLCModelInput(dj.Manual):
    """Table to hold model path if model is being input
    from local disk instead of Spyglass
    """

    definition = """
    dlc_model_local_name : varchar(64)  # Different than dlc_model_name in DLCModelSource... not great
    ---
    project_path         : varchar(255) # Path to project directory
    """
    # TODO: brainstorm what actually lives in this function

    def insert1(self, key, **kwargs):

        project_path = Path(key["project_path"])
        assert project_path.exists(), "project path does not exist"
        super().insert1(key, **kwargs)


@schema
class DLCModelSource(dj.Manual):
    """Table to determine whether model originates from
    upstream DLCModelTraining table, or from local directory
    """

    definition = """
    dlc_model_name : varchar(64)    # User-friendly model name
    ---
    source         : enum ('FromUpstream', 'FromImport')
    """

    class FromImport(dj.Part):
        definition = """
        -> DLCModelSource
        -> DLCModelInput
        """

    class FromUpstream(dj.Part):
        definition = """
        -> DLCModelSource
        -> DLCModelTraining
        """

    @classmethod
    @accepts(None, ("FromUpstream", "FromImport"))
    def insert_entry(cls, key, source: str = "FromUpstream"):

        dlc_model_name = key["dlc_model_name"]
        cls.insert1({"dlc_model_name": dlc_model_name, "source": source})
        part_table = getattr(cls, source)
        part_table.insert1(key)


@schema
class DLCModel(dj.Computed):
    definition = """
    -> DLCModelSource
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
        -> DLCModel
        -> BodyPart
        """

    def make(self, key):
        table_source = DLCModelSource & key().fetch("source")
        source_table = getattr(DLCModelSource, table_source)

        self.insert1(key)
        return None

    @classmethod
    def insert_model(cls, key):
        pass

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
        paramset_name: str = None,
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
        import dlc_reader

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
            # TODO: replace project_path
            "project_path": project_path.relative_to(root_dir).as_posix(),
            "paramset_name": paramset_name,
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


@schema
class DLCModelEvaluation(dj.Computed):
    definition = """
    -> DLCModel
    ---
    train_iterations   : int   # Training iterations
    train_error=null   : float # Train error (px)
    test_error=null    : float # Test error (px)
    p_cutoff=null      : float # p-cutoff used
    train_error_p=null : float # Train error with p-cutoff
    test_error_p=null  : float # Test error with p-cutoff
    """

    def make(self, key):
        """.populate() method will launch evaulation for each unique entry in Model."""
        import csv
        from dlc_reader import dlc_reader
        from deeplabcut import evaluate_network
        from deeplabcut.utils.auxiliaryfunctions import get_evaluation_folder

        dlc_config, project_path, model_prefix, shuffle, trainingsetindex = (
            DLCModel & key
        ).fetch1(
            "config_template",
            "project_path",
            "model_prefix",
            "shuffle",
            "trainingsetindex",
        )

        project_path = find_full_path(get_dlc_root_data_dir(), project_path)
        yml_path, _ = dlc_reader.read_yaml(project_path)

        evaluate_network(
            yml_path,
            Shuffles=[shuffle],  # this needs to be a list
            trainingsetindex=trainingsetindex,
            comparisonbodyparts="all",
        )

        eval_folder = get_evaluation_folder(
            trainFraction=dlc_config["TrainingFraction"][trainingsetindex],
            shuffle=shuffle,
            cfg=dlc_config,
            modelprefix=model_prefix,
        )
        eval_path = project_path / eval_folder
        assert eval_path.exists(), f"Couldn't find evaluation folder:\n{eval_path}"

        eval_csvs = list(eval_path.glob("*csv"))
        max_modified_time = 0
        for eval_csv in eval_csvs:
            modified_time = os.path.getmtime(eval_csv)
            if modified_time > max_modified_time:
                eval_csv_latest = eval_csv
        with open(eval_csv_latest, newline="") as f:
            results = list(csv.DictReader(f, delimiter=","))[0]
        # in testing, test_error_p returned empty string
        self.insert1(
            dict(
                key,
                train_iterations=results["Training iterations:"],
                train_error=results[" Train error(px)"],
                test_error=results[" Test error(px)"],
                p_cutoff=results["p-cutoff used"],
                train_error_p=results["Train error with p-cutoff"],
                test_error_p=results["Test error with p-cutoff"],
            )
        )


def str_to_bool(value) -> bool:
    """Return whether the provided string represents true. Otherwise false."""
    # Due to distutils equivalent depreciation in 3.10
    # Adopted from github.com/PostHog/posthog/blob/master/posthog/utils.py
    if not value:
        return False
    return str(value).lower() in ("y", "yes", "t", "true", "on", "1")

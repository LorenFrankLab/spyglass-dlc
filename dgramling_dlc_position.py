import numpy as np
import pandas as pd
import math
import datajoint as dj
from datetime import datetime
import pynwb
import os
import sys
from itertools import groupby
from operator import itemgetter
import bottleneck as bn
from typing import List, Dict, OrderedDict
from pathlib import Path
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_behav import RawPosition
from dgramling_dlc_pose_estimation import DLCPoseEstimation
from .dgramling_dlc_project import BodyPart

schema = dj.schema('dgramling_dlc_position')

@schema
class DLCSmoothInterpParams(dj.Manual):
    """Parameters for extracting the smoothed head position.
    """

    definition = """
    dlc_si_params_name : varchar(80) # name for this set of parameters
    ---
    params: longblob # dictionary of parameters
    """

    @classmethod
    def insert_params(cls, key, params: dict, skip_duplicates=True):
        cls.insert1(
            {'dlc_si_params_name' : key['params_name'], 
            'params': params},
            skip_duplicates=skip_duplicates)
    
    @classmethod
    def insert_default(cls):
        default_params = {
            'smoothing_params': {
                'smoothing_duration': 0.05,
                },
            'interp_params': {'likelihood_thresh' : 0.95,}
            }
        cls.insert1(
            {'dlc_si_params_name' : 'default', 
            'params': default_params},
            skip_duplicates=True)

@schema
class DLCSmoothInterpSelection(dj.Manual):
    definition="""
    -> DLCPoseEstimation
    -> BodyPart
    -> DLCSmoothInterpParams
    ---

    """

@schema
class DLCSmoothInterp(dj.Computed):
    definition="""
    -> DLCSmoothInterpSelection
    ---
    -> AnalysisNwbfile
    dlc_position_object_id : varchar(80)
    """
    
    def make(self, key):
        key['analysis_file_name'] = AnalysisNwbfile().create(
                                    key['nwb_file_name'])
        # Get labels to smooth from Parameters table
        params = (DLCSmoothInterpParams() & key).fetch1()
        # Get DLC output dataframe
        dlc_df = (DLCPoseEstimation.BodyPart() & key).fetch1_dataframe()[0]
        dlc_df = add_timestamps(dlc_df, key)
        # get interpolated points
        interp_df = interp_pos(dlc_df, **params['interp_params'])
        if 'smoothing_duration' in params['smoothing_params']:
            smoothing_duration = params['smoothing_params'].pop('smoothing_duration')
        dt = np.median(np.diff(dlc_df['time']))
        sampling_rate = 1 / dt
        smooth_df = smooth_pos(
            interp_df,
            smoothing_duration=smoothing_duration,
            sampling_rate=sampling_rate,
            **params['smoothing_params']
            )
        final_df = smooth_df.drop(['likelihood'], axis=1)
        
        # Add dataframe to AnalysisNwbfile
        nwb_analysis_file = AnalysisNwbfile()
        key['dlc_position_object_id'] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=final_df,
        )
        nwb_analysis_file.add(
            nwb_file_name=key['nwb_file_name'],
            analysis_file_name=key['analysis_file_name'])
    
    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (AnalysisNwbfile, 'analysis_file_abs_path'),
                         *attrs, **kwargs)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]['dlc_position'].set_index('time')

def add_timestamps(df: pd.DataFrame, key) -> pd.DataFrame:
    interval_list_name = f'pos {key["epoch"] + 1} valid times'
    raw_pos_df = (
        RawPosition & {'nwb_file_name': key['nwb_file_name'], 
        'interval_list_name' : interval_list_name}
        ).fetch1_dataframe()
    raw_pos_df['time'] = raw_pos_df.index
    raw_pos_df.set_index('video_frame_ind', inplace=True)
    # TODO: do we need to drop indices that don't have a time associated?
    df = df.join(raw_pos_df)
    return df.dropna(subset=['time'])

def interp_pos(dlc_df, **kwargs):
    
    idx = pd.IndexSlice
    subthresh_inds = get_subthresh_inds(
        dlc_df, likelihood_thresh=kwargs.pop('likelihood_thresh'))
    subthresh_spans = get_span_start_stop(subthresh_inds)
    for ind, (span_start, span_stop) in enumerate(subthresh_spans):
        x = [dlc_df['x'].iloc[span_start-1], 
            dlc_df['x'].iloc[span_stop+1]]
        y = [dlc_df['y'].iloc[span_start-1],
            dlc_df['y'].iloc[span_stop+1]]
        span_len = int(span_stop - span_start + 1)
        # TODO: determine if necessary to allow for these parameters
        if 'max_pts_to_interp' in kwargs:
            if span_len > kwargs['max_pts_to_interp']:
                if 'max_cm_to_interp' in kwargs:
                    if (np.linalg.norm(np.array([x[0],y[0]]) - np.array([x[1],y[1]])) 
                        < kwargs['max_cm_to_interp']):
                        change = np.linalg.norm(np.array([x[0],y[0]]) 
                                                - np.array([x[1],y[1]]))
                        print(f'ind: {ind} length: '
                        f'{span_len} interpolated because minimal change:\n {change}cm')
                    else:
                        dlc_df.loc[
                            idx[span_start:span_stop],idx['x']] = np.nan
                        dlc_df.loc[
                            idx[span_start:span_stop],idx['y']] = np.nan
                        print(f'ind: {ind} length: {span_len} '
                        f'not interpolated')
                        continue
        start_time = dlc_df['time'].iloc[span_start]
        stop_time = dlc_df['time'].iloc[span_stop]
        xnew = np.interp(x=dlc_df['time'].iloc[span_start:span_stop+1],
                xp=[start_time, stop_time], fp=[x[0], x[-1]])
        ynew = np.interp(x=dlc_df['time'].iloc[span_start:span_stop+1],
                xp=[start_time, stop_time], fp=[y[0], y[-1]])
        dlc_df.loc[idx[span_start:span_stop],idx['x']] = xnew
        dlc_df.loc[idx[span_start:span_stop],idx['y']] = ynew
    return dlc_df

def get_subthresh_inds(dlc_df: pd.DataFrame, likelihood_thresh: float):
    # Need to get likelihood threshold from kwargs or make it a specified argument
    df_filter = dlc_df['likelihood'] < likelihood_thresh
    sub_thresh_inds = np.where(~np.isnan(
                    dlc_df['likelihood'].where(df_filter)))[0]
    sub_thresh_percent = (len(sub_thresh_inds)/len(dlc_df))*100
    # TODO: add option to return sub_thresh_percent
    return sub_thresh_inds
    
def get_span_start_stop(sub_thresh_inds):

    sub_thresh_spans = []
    for k, g in groupby(enumerate(sub_thresh_inds), lambda x: x[1]-x[0]):
        group = list(map(itemgetter(1),g))
        sub_thresh_spans.append((group[0], group[-1]))
    return sub_thresh_spans

def smooth_pos(interp_df,
    smoothing_duration: float,
    sampling_rate: int,
    **kwargs):
    idx = pd.IndexSlice
    moving_avg_window = int(
        smoothing_duration * sampling_rate)
    xy_arr = interp_df.loc[:,idx[('x','y')]].values
    smoothed_xy_arr = bn.move_mean(xy_arr,
                        window=moving_avg_window,
                        axis=0,min_count=1)
    interp_df.loc[:,idx['x']],interp_df.loc[:,idx['y']] =\
            [*zip(*smoothed_xy_arr.tolist())]
    return interp_df
# DLC Analysis Pipeline
# Run using DEEPLABCUT-rec_to_binaries conda env

import deeplabcut
import numpy as np
import pandas as pd
from pathlib import Path
import os
import cv2
import glob
import math
from tqdm import tqdm as tqdm # redundant?
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, OrderedDict
from operator import itemgetter
from itertools import groupby, combinations
from functools import reduce
import bottleneck as bn
from rec_to_binaries.read_binaries import readTrodesExtractedDataFile

class DLCVideo:
    """
    # Class used to load output from DeepLabCut after a video
    # has been analyzed to perform post-processing

    Attributes
    ----------
    dlc_path : str
        the path of the directory containing the DLC model used to analyze the video
    cfg : structured dict
    """
    def __init__(self, input_filename: str, video_dir:str,
                dlc_model_name: str, dlc_base_path: str,
                labels: List, input_dir: str=None, output_dir: str=None,
                output_filename: str=None):
        """
        Parameters
        ----------
        input_filename : str
            the video file to process in the format of
            'YYYMMDD_SubjectName_EpochNum_SessionNum' (e.g. 20210609_J16_06_r3)
        output_dir : str
            path to save the converted mp4 (e.g. /cumulus/dgramling/converted_videos/)
        input_dir : str
            path where original .h264 video file is located
        output_filename : str
            the name of the overlaid video to output
        dlc_model_name : str
            the name of the DeepLabCut model used to analyze the input video
        dlc_base_path : str
            the path of the directory where `dlc_model_name` resides

        Returns
        -------
        None
        """
        self.labels = labels
        self.epoch = input_filename
        self.save_dir = output_dir #Should probably change this
        self.video_dir = output_dir
        self.output_filename = output_filename
        if input_dir:
            convert_mp4(filename=input_filename+'.1.h264', video_path=input_dir,
                        dest_path=output_dir,videotype='mp4')
        self.dlc_path, self.config = self.get_config(dlc_model_name, dlc_base_path)
        self.dlc_df = self.load_dlc_labels(input_filename=input_filename)
        self.trodes_df, time_df = self.load_trodes_tracking(input_filename=input_filename)
        self.add_time_to_df(time_df)
        self.sub_thresh_dict = self.check_likelihoods(0.95)
        self.get_crop()
        self.interpolate(max_pts_to_interp=5, max_cm_to_interp=6)
        # self.smooth_labels(smoothing_duration=0.05, sampling_rate=50)
        # self.get_head_orientation(method='bisector', use_smooth_labels=True,
        #                         smooth=True,
        #                         label1='redLED_L', label2='redLED_R',
        #                         label3='redLED_C', plot=False)
        

    def get_config(self, dlc_model_name, dlc_base_path) -> "tuple[str, Dict]":
        """
        Parameters
        ----------
        dlc_model_name : str
            the name of the DeepLabCut model used to analyze the input video
        dlc_base_path : str
            the path of the directory where `dlc_model_name` resides

        Returns
        -------
        dlc_path : str
            path to the directory of the DeepLabCut model used to analyze input video
        cfg : ordereddict
            contents of the DLC model's config.yaml read into an ordered dictionary
        """

        dlc_path = dlc_base_path + dlc_model_name + '/'
        cfg = deeplabcut.auxiliaryfunctions.read_config(dlc_path + 'config.yaml')
        return dlc_path, cfg
    
    def load_dlc_labels(self, input_filename: str) -> pd.DataFrame:
        """
        Get positions of labels for each frame of the recording using :meth:`pd.read_hdf`

        Parameters
        ----------
        input_filename : str
            the video file to process in the format of
            'YYYMMDD_SubjectName_EpochNum_SessionNum' (e.g. 20210609_J16_06_r3)

        Returns
        -------
        dlc_df : pd.DataFrame
            Pandas DataFrame containing the coordinates and likelihood for each label
            on each video frame with columns
        """

        video_file = input_filename
        h5_file = glob.glob(self.dlc_path + "*" + video_file + "*.h5")[0]
        dlc_df = pd.read_hdf(h5_file)
        # greenLED = dlc_df[dlc_df.keys()[0][0]]['greenLED']
        # redLED_L = dlc_df[dlc_df.keys()[0][0]]['redLED_L']
        # redLED_C = dlc_df[dlc_df.keys()[0][0]]['redLED_C']
        # redLED_R = dlc_df[dlc_df.keys()[0][0]]['redLED_R']
        dlc_df = dlc_df[dlc_df.keys()[0][0]]
        return dlc_df
    
    def load_trodes_tracking(self, input_filename: str,
                            online_tracking: bool=False) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """
        Get positions of red and green LED as identified by Trodes Tracking (optional)
        Load in the cameraHWSync file from Trodes to match timestamps to video frames

        Parameters
        ----------
        input_filename : str
            the video file to process in the format of
            'YYYMMDD_SubjectName_EpochNum_SessionNum' (e.g. 20210609_J16_06_r3)
        online_tracking : bool
            whether the Trodes tracking was performed online or offline (default is False)

        Returns
        -------
        trodes_df : pd.DataFrame
            Pandas DataFrame containing the coordinates for the 
            red and green LED as identified by Trodes tracking
            (either online or offline)
        time_df : pd.DataFrame
            Pandas DataFrame to match timestamps to video frames.
            Columns include `PosTimestamp`, `HWframeCount`, `HWTimestamp` 
        """

        video_file = input_filename
        day = video_file.split('_')[0]
        trodes_filepath = (f'/cumulus/jguidera/J16/{day}/{video_file}/')
        self.trodes_labels = ['greenX', 'greenY', 'redX', 'redY']
        if online_tracking:
            trodes_filename = video_file + '.1.tracking.pos_online'   
        else: 
            trodes_filename = video_file + '.1.tracking.pos_offline'
        trodes_data = readTrodesExtractedDataFile(
            (trodes_filepath + trodes_filename))
        trodes_df = pd.DataFrame.from_records(trodes_data['data'])
        trodes_df.rename(
            columns={'xloc': 'greenX', 'yloc': 'greenY',
            'xloc2': 'redX', 'yloc2': 'redY'}, inplace=True)
        time_data = readTrodesExtractedDataFile((
            trodes_filepath + video_file + '.1.videoTimeStamps.cameraHWSync'))
        time_df = pd.DataFrame.from_records(time_data['data'])
        return trodes_df, time_df
    
    def add_time_to_df(self, time_df: pd.DataFrame):
        """
        Adds timestamps from Trodes cameraHWSync to dlc_df

        Parameters
        ----------
        time_df : pd.DataFrame
            dataframe from Trodes cameraHWSync
        
        Returns
        -------
        None
        """
        # Convert HWTimestamps from trodes to pandas datetime
        if 'Datetime' in self.dlc_df:
            del self.dlc_df['Datetime']
        if 'HWTimestamp' in self.dlc_df:
            del self.dlc_df['HWTimestamp']
        self.dlc_df['HWTimestamp'] = time_df['HWTimestamp'].copy()
        self.dlc_df['Datetime'] = pd.DataFrame(np.asarray(
            self.dlc_df['HWTimestamp'], dtype='datetime64[ns]'))
        self.dlc_df['Datetime'] = self.dlc_df['Datetime'].dt.tz_localize(
                                                                    'UTC')
        self.dlc_df['Datetime'] = self.dlc_df['Datetime'].dt.tz_convert(
                                                                'US/Pacific')

    def check_likelihoods(self, likelihood_thresh: float=0.95,
                            labels: List=None, plot: bool=False) -> Dict:
        """
        Get positions of red and green LED as identified by Trodes Tracking
        (optional)
        Load in the cameraHWSync file from Trodes to match timestamps 
        to video frames

        Parameters
        ----------
        dlc_df : pd.DataFrame
            Pandas DataFrame containing the coordinates
            and likelihood for each label on each video frame with columns
        likelihood_thresh : float
            Threshold to define bad/good labels based on likelihood
            (default is 0.95)
        labels : List
            list of strings for each label of interest in the DLC model
            (e.g. ['redLED_L', 'greenLED'], default is None)
        plot : bool
            True if want to plot the distribution of indices with low 
            likelihood that overlap across different labels

        Returns
        -------
        sub_thresh_dict : Dict
            Dictionary containing spans of indices where label
            is below specified likelihood threshold
        """
        if not labels:
            labels = self.labels
        df_filter = {label:self.dlc_df[label]['likelihood'] 
                    < likelihood_thresh for label in labels}
        sub_thresh_dict = {col : {'inds' : np.where(~np.isnan(
                        self.dlc_df[col]['likelihood'].where(df_filter[col])))[0],
                        'percent' : (len(np.where(
                        ~np.isnan(self.dlc_df[col]['likelihood'].where(
                        df_filter[col])))[0])/len(self.dlc_df))*100,
                        'span_length' : [], 'span_inds' : []} for col in labels}
        led_combs = list(combinations(sub_thresh_dict, 2))
        led_below_overlap = {f'{led1}_{led2}':np.intersect1d(
                sub_thresh_dict[led1]['inds'], sub_thresh_dict[led2]['inds'])
                for led1, led2 in led_combs}
        if plot:
            self.plot_spans(led_below_overlap)
        print(f'percent of frames below {likelihood_thresh} '
            f'likelihood threshold for {labels[0]}: '
            f'{sub_thresh_dict[labels[0]]["percent"]} and {labels[1]}\n'
            f': {sub_thresh_dict[labels[1]]["percent"]}\n{labels[2]}: '
            f'{sub_thresh_dict[labels[2]]["percent"]}\n{labels[3]}: '
            f'{sub_thresh_dict[labels[3]]["percent"]}')
        sub_thresh_dict = self.check_for_spans_below_thresh(sub_thresh_dict=sub_thresh_dict)
        return sub_thresh_dict
    
    def check_for_spans_below_thresh(self, sub_thresh_dict: Dict):
        """
        Add the start and stop indices of a span below likelihood threshold
        as well as the length of each span

        Parameters
        ----------
        sub_thresh_dict : Dict
            Dictionary containing indices where label
            is below likelihood threshold

        Returns
        -------
        sub_thresh_dict : Dict
            updated sub_thresh_dict with span lengths
            and start and stop indices
        """
        for key in sub_thresh_dict.keys():
            sub_thresh_dict[key]['span_inds'] = []
            sub_thresh_dict[key]['span_length'] = []
            for k, g in groupby(enumerate(sub_thresh_dict[key]['inds']), lambda x: x[1]-x[0]):
                group = list(map(itemgetter(1),g))
                sub_thresh_dict[key]['span_inds'].append((group[0], group[-1]))
                sub_thresh_dict[key]['span_length'].append(sum(1 for _ in group))
        return sub_thresh_dict
    
    def plot_spans(self, led_below_overlap: dict, sub_thresh_dict: Dict=None):
        # TODO Make this a time series plot showing a line where indices are below likelihood
        # similar to interval list plot
        if sub_thresh_dict:
            fig, ax = plt.subplots(nrows=2, figsize=(7.2, 9))
        else:
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
        x_pts = np.arange(0.5,len(led_below_overlap) + 0.5)
        pts_overlap = [len(led_below_overlap[key]) for key in led_below_overlap.keys()]
        if sub_thresh_dict:
            ax[0].bar(x_pts, pts_overlap, width=0.5)
            for ind, label in enumerate(sub_thresh_dict.keys()):
                ax[1].plot(sub_thresh_dict[label]['inds'],
                np.full_like(sub_thresh_dict[label]['inds'], int(ind)),
                c=f'C{ind}')
        else:
            ax.bar(x_pts, pts_overlap, width=0.5)
        # fig.savefig()
    
    def get_crop(self):
        """
        Get the pixel values of the crop used during 
        the DeepLabCut model creation
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        
        Other
        -----
        crop_int : List
            List of integers representing the crop of the input video
            ([x1, x2, y1, y2])
        """
        crop = self.config['video_sets'][
            list(self.config['video_sets']._keys())[0]]['crop']
        self.crop = [int(val) for val in crop.replace(' ','').split(',')]
    
    def convert_pixels_to_cm(self, environment: str):
        """
        Returns conversion from meters to pixels for specified environment

        Parameters
        ----------
        environment : str
            string representing environment where epoch occurred
            (e.g. 'SA', 'LH', 'Sleep')
        
        Returns
        -------
        None
        
        Other
        -----
        cm_per_pixel : float
            value to convert from pixels in the video to centimeters 
        """
        conversion_dict = dict(meters_per_pixel_SA = 0.003427,
            meters_per_pixel_LH = 0.002740,
            meters_per_pixel_RH = 0.002740,
            meters_per_pixel_Sleep = 0.001399,
            meters_per_pixel_Home = 0.000842)
        conversion_val = [val for key,val in conversion_dict.items()
                            if environment in key]
        self.cm_per_pixel = conversion_val * 100

    def interpolate(self, max_pts_to_interp: int=5, max_cm_to_interp: float=0):
        """
        Interpolate over low likelihood spans per label

        Parameters
        ----------
        max_pts_to_interp : int
            max length of consecutive frames to interpolate over
        max_cm_to_interp : float
            (Default is 0), If set, will disregard max_pts_to_interp if 
            distance from the start of low likelihood span to the stop 
            is less than set value
        
        Returns
        -------
        None

        Others
        ------
        self.interp_df : pd.DataFrame
            DataFrame with interpolated values
        """
        interp_df = self.dlc_df.copy()
        idx = pd.IndexSlice
        for key in self.sub_thresh_dict.keys():
            interp_df[key, 'interpolated'] = np.full_like(self.dlc_df[key,'x'], False, dtype=bool)
            for ind, (span_start, span_stop) in enumerate(self.sub_thresh_dict[key]['span_inds']):
                x = [self.dlc_df[key,'x'].iloc[span_start-1], 
                    self.dlc_df[key,'x'].iloc[span_stop+1]]
                y = [self.dlc_df[key,'y'].iloc[span_start-1],
                    self.dlc_df[key,'y'].iloc[span_stop+1]]
                if self.sub_thresh_dict[key]['span_length'][ind] > max_pts_to_interp:
                    if (np.linalg.norm(np.array([x[0],y[0]]) - np.array([x[1],y[1]])) 
                        < max_cm_to_interp):
                        change = np.linalg.norm(np.array([x[0],y[0]]) 
                                                - np.array([x[1],y[1]]))
                        print(f'{key}, ind: {ind} length: '
                        f'{self.sub_thresh_dict[key]["span_length"][ind]} '
                        f'interpolated because minimal change:\n {change}cm')
                        pass
                    else:
                        print(f'{key}, ind: {ind} length: '
                        f'{self.sub_thresh_dict[key]["span_length"][ind]} '
                        f'not interpolated')
                        continue
                start_time = self.dlc_df['HWTimestamp'].iloc[span_start]
                stop_time = self.dlc_df['HWTimestamp'].iloc[span_stop]
                xnew = np.interp(x=self.dlc_df['HWTimestamp'].iloc[span_start:span_stop+1],
                        xp=[start_time, stop_time], fp=[x[0], x[-1]])
                ynew = np.interp(x=self.dlc_df['HWTimestamp'].iloc[span_start:span_stop+1],
                        xp=[start_time, stop_time], fp=[y[0], y[-1]])
                interp_df.loc[idx[span_start:span_stop],idx[key, 'x']] = xnew
                interp_df.loc[idx[span_start:span_stop],idx[key, 'y']] = ynew
                interp_df.loc[idx[span_start:span_stop],idx[key, 'interpolated']] = True
                print(f'{key}, ind: {ind} length: '
                f'{self.sub_thresh_dict[key]["span_length"][ind]} '
                f'interpolated')
                self.interp_df = interp_df
    
    def smooth_labels(self, smoothing_duration: float=0.05, 
                        sampling_rate: int=50, labels: List=None):
        """
        Applies smoothing to each label, if interpolated True, applies to interp_df

        Parameters
        ----------
        smoothing_duration : float
            time in sec over which to smooth (default is 0.05)
        sampling_rate : int
            frames per second of input video file (default is 50)
        labels : List
            list of label names (default is None)
        """
        if not labels:
            labels = self.labels
        idx = pd.IndexSlice
        moving_avg_window = int(
            smoothing_duration * sampling_rate)
        for label in labels:
            xy_label_arr = self.interp_df.loc[:,idx[label,('x','y')]].values
            smoothed_xy_arr = bn.move_mean(xy_label_arr,
                                window=moving_avg_window,
                                axis=0,min_count=1)
            self.interp_df.loc[:,idx[f'{label}_smoothed','x']],\
                self.interp_df.loc[:,idx[f'{label}_smoothed','y']] =\
                    [*zip(*smoothed_xy_arr.tolist())]
    
    def get_head_orientation(self, method: str='bisector', use_smooth_labels: bool=True,
                             smooth: bool=True, head_orient_smoothing_std_dev: float=0.001,
                             sampling_rate: int=50, label1: str=None,
                             label2: str=None, label3: str=None, plot=True):
        """
        Calculates head orientation using either a bisector of 2 labels
        or the vector between the green and red LEDs
        
        Parameters
        ----------
        method : str
            (Default is green-red) Options are:

            -   'green-red' : uses the vector from the red LED to the green LED
                to determine head orientation
            
            -   'bisector' : find the bisector of the left and right red LEDs
                and use that as the head orientation
        use_smooth_labels : bool
            if True uses smoothed position of labels to calculate head direction
        smooth : bool
            if True smooths the head orientation using a gaussian kernel
            (Default is True)
        head_orient_smoothing_std_dev : float
            time in sec over which to smooth (default is 0.001)
        sampling_rate : int
            frames per second of input video file (default is 50)
        label1 : str
            label to use for green LED in green-red method
            label to use for LED 1 in bisector method
        label2 : str
            label to use for red lED in green-red method
            label to use for LED 2 in bisector method
        label3 : str
            label to use in bisector method to determine head direction
            if y vector equals 0, should be behind the two labels 
            used for finding bisector
        plot : bool
            if True plots head direction in polar space
            (Default is False)
    
        Notes
        -----
        Updates self.interp_df inplace
        """
        # TODO: add support to get labels from self.labels based off method selected

        if not label1:
            raise Exception(f'No values for label1, '
                        f'please pass arguments for label1 and label2')
        if not label2:
            raise Exception(f'No values for label2, '
                        f'please pass arguments for label1 and label2')
        if use_smooth_labels:
            if 'smooth' in label1:
                pass
            else:
                label1 = label1 + '_smoothed'
            if 'smooth' in label2:
                pass
            else:
                label2 = label2 + '_smoothed'
        method = method.lower()
        assert method in ['bisector', 'green-red']
        if method == 'green-red':
            self._head_orientation = 'green-red'
            self.interp_df['head_orientation'] = np.arctan2(
                (self.interp_df[label1, 'y'] 
                - self.interp_df[label2, 'y']),
                (self.interp_df[label1, 'x'] 
                - self.interp_df[label2, 'x']))
        if method == 'bisector':
            self._head_orientation = 'bisector'
            x_vec = (self.interp_df[label1,'x'] - self.interp_df[label2,'x'])
            y_vec = (self.interp_df[label1,'y'] - self.interp_df[label2,'y'])
            y_zeros = np.where(y_vec == 0)[0]
            y_vec[y_zeros] = np.nan
            length = np.sqrt(y_vec*y_vec + x_vec*x_vec)
            norm = np.array([-y_vec/length, x_vec/length])
            head_orient_init = np.arctan2(norm[1], norm[0])
            if len(y_zeros) > 0:
                if not label3:
                    raise Exception(f'Unable to find head direction for {len(y_zeros)} frames'
                                    f'without `label3`. Please pass label3')
                for ind in y_zeros:
                    if ((self.interp_df[label3,'y'].iloc[ind]
                        > self.interp_df[label1,'y'].iloc[ind]) &
                        (self.interp_df[label3,'y'].iloc[ind] 
                        > self.interp_df[label2,'y'].iloc[ind])):
                        head_orient_init[ind] = (np.pi / 2)
                    elif ((self.interp_df[label3,'y'].iloc[ind] 
                        < self.interp_df[label1,'y'].iloc[ind]) &
                        (self.interp_df[label3,'y'].iloc[ind] 
                        < self.interp_df[label2,'y'].iloc[ind])):
                        head_orient_init[ind] = -(np.pi / 2)
                    else:
                        raise Exception('Cannot determine head direction from bisector')
            self.interp_df['head_orientation'] = head_orient_init
        if smooth:
            # Smooth head orientation
            from position_tools.core import gaussian_smooth
            head_orient = self.interp_df['head_orientation'].to_numpy()
            is_nan = np.isnan(head_orient)

            # Unwrap orientation before smoothing
            head_orient[~is_nan] = np.unwrap(head_orient[~is_nan])
            head_orient[~is_nan] = gaussian_smooth(
                head_orient[~is_nan], head_orient_smoothing_std_dev,
                sampling_rate, axis=0, truncate=8)
            # convert back to between -pi and pi
            head_orient[~is_nan] = np.angle(
                np.exp(1j * head_orient[~is_nan]))
            self.interp_df['head_orient_smoothed'] = head_orient
        if plot:
            if smooth:
                self.polar_hist([self.interp_df['head_orient_smoothed']],
                                labels=[method + '_smoothed'],
                                title='Smoothed Head Direction')
            else:
                self.polar_hist([self.interp_df['head_orientation']],
                                labels=[method],
                                title='Head Direction')

    def polar_hist(self, dist_list: List, norm: bool=True, n_bins: int=1000,
                    labels: List=None, title: str=None, save: bool=True):
        bottom = 0
        max_height = 10
        num_bin = n_bins
        theta = np.linspace(-np.pi, np.pi, num_bin+1)
        bin_centers = 0.5*(theta[:-1]+theta[1:])
        fig, ax = plt.subplots(figsize=(15,15), subplot_kw=dict(projection='polar'))
        for ind, dist in enumerate(dist_list):
            radii, _ = np.histogram(dist, bins=theta)
            if norm:
                radiib = radii/np.max(radii) * max_height
            else:
                radiib = radii
            bin_widths = theta[1:]-theta[:-1]
            ax.set_theta_direction(-1)
            ax.set_theta_offset(np.pi / 2.0)
            bars = ax.bar(bin_centers, radiib, width=bin_widths, bottom=bottom,
                        alpha=0.6, color=f'C{ind}', label=labels[ind])
        ax.set_title(title)
        ax.legend()
        if save:
            init_dir = os.getcwd()
            os.chdir(self.save_dir)
            fig.savefig(f'{self.epoch}_{labels}_head_direction.png', dpi=300)
            os.chdir(init_dir)
        else:
            plt.show()
    
    def get_centroid(self, green_label: str, red_label: str,
                    use_smoothed: bool=True, comp: str=None,
                    trodes_df: pd.DataFrame=None):
        """
        Get centroid of labels

        Parameters
        ----------

        Returns
        -------
        centroid_dict
        """
        # TODO: figure out how to handle different combinations of labels to find centroid
        # Current thinking is check for labels = 2 or labels = 4 and if labels =2 only use
        # green and red LED and if not interpolated or high likelihood then NaN
        # Basically - this needs to generalize
        labels = self.labels.copy()
        if use_smoothed:
            substr = '_smoothed'
            low_inds_dict = {label + substr: self.interp_df.index[
                self.interp_df[label, 'likelihood'] < 0.95].to_numpy()
                for label in labels}
            labels = [label + substr for label in labels]
        else:
            low_inds_dict = {label: self.interp_df.index[
                self.interp_df[label, 'likelihood'] < 0.95].to_numpy()
                for label in labels}
        interp_centroid = []
        if comp:
            raise NotImplementedError('comp is not currently implemented')
            if comp in ['raw_smooth', 'raw_interp', 'interp_smooth', 'trodes_smooth']:
                comp1, comp2 = comp.split('_')
                comp1_df = self.str_to_df(comp1)
                comp2_df = self.str_to_df(comp2)
                if comp == 'raw_smooth':
                    comp1_labels = self.labels
                    comp2_labels = [label + '_smoothed' for label in self.labels]
                if comp == 'raw_interp':
                    comp1_labels = self.labels
                    comp2_labels = self.labels
                if comp == 'interp_smooth':
                    comp1_labels = self.labels
                    comp2_labels = [label + '_smoothed' for label in self.labels]
                if comp == 'trodes_smooth':
                    comp1_labels = self.trodes_labels
                    comp2_labels = [label + '_smoothed' for label in self.labels]
            else:
                raise Exception(f'comparison: {comp} not a valid option.')
            centroid_dict = {comp1:{'x':[], 'y':[]}, comp2:{'x':[], 'y':[]}}
            centroid_diff = []
            comp1_centroid = []
            comp2_centroid = []
            for (idx1, comp1_row), (idx1, comp2_row) in zip(comp1_df.iterrows(),
                                                            comp2_df.iterrows()):
                if 'raw' in comp:
                #### ----dlc_centroid without interpolation---- ####
                    if idx1 in low_green_inds:
                        if not idx1 in low_redC_inds:
                            dlc_centroid = (comp1_row['redLED_C','x'], comp1_row['redLED_C','y'])
                        elif not idx1 in low_redR_inds:
                            if not idx1 in low_redL_inds:
                                midpoint = ((comp1_row['redLED_L','x'] + comp1_row['redLED_R','x']) / 2,
                                            (comp1_row['redLED_L','y'] + comp1_row['redLED_R','y']) / 2)
                                dlc_centroid = midpoint
                            else:
                                dlc_centroid = (comp1_row['redLED_R','x'],
                                                comp1_row['redLED_R','y'])
                        else: # Theoretically no green, red center, or red right
                            if not idx1 in low_redL_inds:
                                dlc_centroid = (comp1_row['redLED_L','x'],
                                                comp1_row['redLED_L','y'])
                            dlc_centroid = (np.nan, np.nan)
                    elif idx1 in low_redC_inds:
                        # check if the green led has high likelihood
                #         if not idx1 in low_green_inds: # shouldn't need this because of parent elif nature
                            # check if the green led and right red led have high likelihood
                        if not idx1 in low_redR_inds:
                            # check if the green led and right and left red led have high likelihood
                            if not idx1 in low_redL_inds:
                                midpoint = ((comp1_row['redLED_L','x'] + comp1_row['redLED_R','x']) / 2,
                                            (comp1_row['redLED_L','y'] + comp1_row['redLED_R','y']) / 2)
                                dlc_centroid = ((midpoint[0] + comp1_row['greenLED','x']) / 2,
                                                (midpoint[1] + comp1_row['greenLED','y']) / 2)
                            # if the green and right red have high likelihood, but not right red led
                            else:
                                dlc_centroid = ((comp1_row['redLED_R','x'] + comp1_row['greenLED','x']) / 2,
                                                (comp1_row['redLED_R','y'] + comp1_row['greenLED','y']) / 2)
                        # if the green led and left red led have high likelihood, but not the red right led
                        elif not idx1 in low_redR_inds:
                            dlc_centroid = ((comp1_row['redLED_R','x'] + comp1_row['greenLED','x']) / 2,
                                            (comp1_row['redLED_R','y'] + comp1_row['greenLED','y']) / 2)
                        else:
                            dlc_centroid = (np.nan, np.nan)
                    else:
                        dlc_centroid = ((comp1_row['redLED_C','x'] + comp1_row['greenLED','x']) / 2,
                                        (comp1_row['redLED_C','y'] + comp1_row['greenLED','y']) / 2)
                # TODO: LOL AT THIS MESS
                if 'trodes' in comp1:
                    # Trodes centroid
                    trodes_centroid = ((comp1_row['greenX'] + comp1_row['redX']) / 2,
                                    (comp1_row['greenY'] + comp1_row['redY']) / 2)
                if 'trodes' in comp2:
                    # Trodes centroid
                    trodes_centroid = ((comp2_row['greenX'] + comp2_row['redX']) / 2,
                                    (comp2_row['greenY'] + comp2_row['redY']) / 2)
                # This won't work either
                centroid_diff.append(math.dist(dlc_centroid, interp_centroid))
        centroid_x = []
        centroid_y = []
        for (idx1, interp_row) in self.interp_df.iterrows():
            # TODO: this feels messy, clean-up
            green_label = [label for label in labels if 'green' in label][0]
            red_label = [label for label in labels if ('red' in label) & ('C' in label)][0]
            labels = [label for label in labels if label not in (green_label, red_label)]
            alt_label1 = labels[0]
            alt_label2 = labels[1]
            if use_smoothed:
                raw_green = green_label.removesuffix(substr)
                raw_red = red_label.removesuffix(substr)
            
            #### ----dlc_centroid with interpolation---- ####
            if idx1 in low_inds_dict[green_label]:
                if interp_row[raw_green, 'interpolated'] == True:
                    if not idx1 in low_inds_dict[red_label]:
                        interp_centroid = ((interp_row[red_label,'x']
                                            + interp_row[green_label,'x']) / 2,
                                            (interp_row[red_label,'y'] 
                                            + interp_row[green_label,'y']) / 2)
                    elif interp_row[raw_red, 'interpolated'] == True:
                        interp_centroid = ((interp_row[red_label,'x'] 
                                            + interp_row[green_label,'x']) / 2,
                                            (interp_row[red_label,'y'] 
                                            + interp_row[green_label,'y']) / 2)
                    elif ((not idx1 in low_inds_dict[alt_label1]) & 
                        (not idx1 in low_inds_dict[alt_label2])):
                        midpoint = ((interp_row[alt_label1,'x'] + interp_row[alt_label2,'x']) / 2,
                                    (interp_row[alt_label1,'y'] + interp_row[alt_label2,'y']) / 2)
                        interp_centroid = ((midpoint[0] + interp_row[green_label,'x']) / 2,
                                        (midpoint[1] + interp_row[green_label,'y']) / 2)          
                elif not idx1 in low_inds_dict[red_label]:
                    interp_centroid = (interp_row[red_label,'x'],
                                    interp_row[red_label,'y'])
                elif interp_row['redLED_C', 'interpolated'] == True:
                    interp_centroid = (interp_row[red_label,'x'],
                                    interp_row[red_label,'y'])
                elif ((not idx1 in low_inds_dict[alt_label1]) & 
                    (not idx1 in low_inds_dict[alt_label2])):
                    midpoint = ((interp_row[alt_label1,'x'] + interp_row[alt_label2,'x']) / 2,
                                (interp_row[alt_label1,'y'] + interp_row[alt_label2,'y']) / 2)
                    interp_centroid = midpoint
                else: # Theoretically no green, red center, or red right
                    if not idx1 in low_inds_dict[alt_label1]:
                        interp_centroid = (interp_row[alt_label1,'x'],
                                        interp_row[alt_label1,'y'])
                    elif not idx1 in low_inds_dict[alt_label2]:
                        interp_centroid = (interp_row[alt_label2,'x'],
                                        interp_row[alt_label2,'y'])
                    else:
                        interp_centroid = (np.nan, np.nan)
                        print(f'{idx1} not interpolated with green and red '
                        f'(L, C, R) LED likelihoods: '
                        f'{[interp_row[key.removesuffix(substr), "likelihood"] for key in labels]}\n'
                        f'and interpolation status: '
                        f'{[interp_row[key.removesuffix(substr), "interpolated"] for key in labels]}')
            # Good green LED, but bad Red Center LED
            elif idx1 in low_inds_dict[red_label]:
                if interp_row[raw_red, 'interpolated'] == True:
                    interp_centroid = ((interp_row[red_label,'x'] + interp_row[green_label,'x']) / 2,
                                    (interp_row[red_label,'y'] + interp_row[green_label,'y']) / 2)
                # check if the green led and right red led have high likelihood
                elif not idx1 in low_inds_dict[alt_label2]:
                    # check if the green led and right and left red led have high likelihood
                    if not idx1 in low_inds_dict[alt_label1]:
                        midpoint = ((interp_row[alt_label1,'x'] + interp_row[alt_label2,'x']) / 2,
                                    (interp_row[alt_label1,'y'] + interp_row[alt_label2,'y']) / 2)
                        interp_centroid = ((midpoint[0] + interp_row[green_label,'x']) / 2,
                                        (midpoint[1] + interp_row[green_label,'y']) / 2)
                    # if the green and right red have high likelihood, but not right red led
                    else:
                        interp_centroid = ((interp_row[alt_label2,'x'] + interp_row[green_label,'x']) / 2,
                                        (interp_row[alt_label2,'y'] + interp_row[green_label,'y']) / 2)
                # if the green led and left red led have high likelihood, but not the red right led
                elif not idx1 in low_inds_dict[alt_label2]:
                    interp_centroid = ((interp_row[alt_label2,'x'] + interp_row[green_label,'x']) / 2,
                                    (interp_row[alt_label2,'y'] + interp_row[green_label,'y']) / 2)
                else:
                    interp_centroid = (np.nan, np.nan)
                    print(f'{idx1} not interpolated with green and red (L, C, R) LED likelihoods: '
                        f'{[interp_row[key.removesuffix(substr), "likelihood"] for key in labels]}\n'
                        f'and interpolation status: '
                        f'{[interp_row[key.removesuffix(substr), "interpolated"] for key in labels]}')
            else:
                interp_centroid = ((interp_row[red_label,'x'] + interp_row[green_label,'x']) / 2,
                                (interp_row[red_label,'y'] + interp_row[green_label,'y']) / 2)
            
            centroid_x.append(interp_centroid[0])
            centroid_y.append(interp_centroid[1])
        centroid_dict = {}
        centroid_dict['x'] = np.array(centroid_x)
        centroid_dict['y'] = np.array(centroid_y)

    def str_to_df(self, df_name):
        if df_name == 'trodes':
            return self.trodes_df
        if df_name == 'raw':
            return self.dlc_df
        if df_name == 'interp':
            return self.interp_df
        if df_name == 'smooth':
            return self.interp_df
    
    def make_overlaid_video(self, output_filename:str, smooth: bool=True, input_filename:str=None,
                            sampling_frequency: int=50, video_slowdown:int=1, 
                            crop: bool=True, crop_int: List=[], labels_to_plot: List=None,
                            plot_centroid: bool=True):
        """
        Create video with various positions indicators overlaid on behavior recording video
        """
        import matplotlib.animation as animation
        import matplotlib.font_manager as fm
        time_slice = []
        # TODO: make variable for video output directory
        os.chdir('/cumulus/dgramling/deeplabcut/Overlaid_videos/')
        if not input_filename:
            input_filename = self.epoch
        if smooth:
            movie_name = (f'{output_filename}_dlc_overlay_smoothed.mp4')
        else:
            movie_name = (f'{output_filename}_dlc_overlay.mp4')
        vmax = 0.07 #?
        if not labels_to_plot:
            labels = self.labels
        else:
            labels = labels_to_plot
        # Probably don't need
        # if plot_centroid:
        #     labels.append('centroid')
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        fps = sampling_frequency // video_slowdown
        writer = Writer(fps=fps, bitrate=-1)
        
        # Set up data
        head_orientation = np.asarray(self.interp_df['head_orient_smoothed'])
        window_size = 501
        window_ind = np.arange(window_size) - window_size // 2

        # Get video frames
        # TODO: create logic to get to video directory
        video_filename = (f'{self.video_dir}/{input_filename}.mp4')
        video = cv2.VideoCapture(video_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (int(video.get(3)), int(video.get(4)))
        frame_rate = video.get(5)
        n_frames = len(self.interp_df['HWTimestamp'])
        ret,frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if crop:
            frame = frame[crop_int[2]:crop_int[3], crop_int[0]:crop_int[1]].copy()
            crop_offset_x = crop_int[0]
            crop_offset_y = crop_int[2]
        with plt.style.context("dark_background"):
            # Set up plots
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [8, 1]},
                constrained_layout=False,
            )

            axes[0].tick_params(colors='white', which='both')
            axes[0].spines['bottom'].set_color('white')
            axes[0].spines['left'].set_color('white')
            image = axes[0].imshow(frame)
            # TODO: turn into dictionary to handle flexible number of labels
            label_plot_dict = {key:{'position': None, 'likelihood': None} for key in labels}
            green_position_dot = axes[0].scatter([], [], s=2, zorder=102, color='#29ff3e',
                                        label='green position', animated=True, alpha=0.6)
            redL_position_dot = axes[0].scatter([], [], s=2, zorder=102, color='#ff0073',
                                        label='redL position', animated=True, alpha=0.6)
            redC_position_dot = axes[0].scatter([], [], s=2, zorder=102, color='#ff291a',
                                        label='redC position', animated=True, alpha=0.6)
            redR_position_dot = axes[0].scatter([], [], s=2, zorder=102, color='#1e2cff',
                                        label='redR position', animated=True, alpha=0.6)
            # TODO: need another flag for whether or not to plot multiple centroids
            # dlc_position_dot = axes[0].scatter([], [], s=5, zorder=102, color='#b045f3',
            #                             label='DLC position', animated=True, alpha=0.6)
            if plot_centroid:
                if smooth:
                    centroid_position_dot = axes[0].scatter([], [], s=5, zorder=102, color='#ffe91a',
                                                label='smoothed, interpolated position', animated=True, alpha=0.6)
                else:
                    centroid_position_dot = axes[0].scatter([], [], s=5, zorder=102, color='#ffe91a',
                                                label='interpolated position', animated=True, alpha=0.6)
            if self._head_orientation == 'green-red':
                (head_orient_line,) = axes[0].plot(
                    [], [], color='green', linewidth=1, animated=True, label='Green-Red Smoothed')
            if self._head_orientation == 'bisector':
                (head_orient_line,) = axes[0].plot(
                    [], [], color='cyan', linewidth=1, animated=True, label='Bisector Smoothed')
            axes[0].set_xlabel('')
            axes[0].set_ylabel('')
            ratio = frame_size[1] / frame_size[0]
            if crop:
                ratio = (crop_int[3] - crop_int[2])/(crop_int[1] - crop_int[0]) #630/380
            x_left, x_right = axes[0].get_xlim()
            y_low, y_high = axes[0].get_ylim()
            axes[0].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
            axes[0].spines['top'].set_color('black')
            axes[0].spines['right'].set_color('black')
            time_delta = pd.Timedelta(self.interp_df['Datetime'].iloc[0] 
                                    - self.interp_df['Datetime'].iloc[0]).total_seconds()
            axes[0].legend(loc='lower right', fontsize=4)
            title = axes[0].set_title(
                f'time = {time_delta:3.4f}s\n frame = {self.interp_df.index[0]}',
                fontsize=8,)
            fontprops = fm.FontProperties(size=12)
            axes[0].axis('off')
            # Make dict for likelihood plots as well
            (redL_likelihood,) = axes[1].plot([], [], color='yellow', linewidth=1,
                                                    animated=True, clip_on=False,
                                                    label='red_left')
            (redR_likelihood,) = axes[1].plot([], [], color='blue', linewidth=1,
                                                    animated=True, clip_on=False,
                                                    label='red_right')
            (green_likelihood,) = axes[1].plot([], [], color='green', linewidth=1,
                                                    animated=True, clip_on=False,
                                                    label='green')
            (redC_likelihood,) = axes[1].plot([], [], color='red', linewidth=1,
                                                    animated=True, clip_on=False,
                                                    label='red_center')
            axes[1].set_ylim((0.0, 1))
            axes[1].set_xlim((window_ind[0] / sampling_frequency,
                            window_ind[-1] / sampling_frequency))
            axes[1].set_xlabel('Time [s]')
            axes[1].set_ylabel('Likelihood')
            axes[1].set_facecolor("black")
            axes[1].spines['top'].set_color('black')
            axes[1].spines['right'].set_color('black')
            axes[1].legend(loc='upper right', fontsize=4)

            n_frames = int(self.interp_df.shape[0] / 10)
            progress_bar = tqdm()
            progress_bar.reset(total=n_frames)

            def _update_plot(time_ind):
                start_ind = max(0, time_ind - 5)
                time_slice = slice(start_ind, time_ind)
                ret, frame = video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if crop:
                        frame = frame[crop_int[2]:crop_int[3], crop_int[0]:crop_int[1]].copy()
                    image.set_array(frame)
                if plot_centroid:
                    interp_centroid_data = np.hstack((centroid_dict['interp']['x'][time_ind,np.newaxis],
                                                    centroid_dict['interp']['y'][time_ind,np.newaxis]))
                # dlc_centroid_data = np.hstack((centroid_dict['dlc']['x'][time_ind,np.newaxis],
                #                             centroid_dict['dlc']['y'][time_ind,np.newaxis]))
                if smooth:
                    greenLED_data = np.hstack((interp_df['greenLED_smoothed','x'][time_ind,np.newaxis],
                                            interp_df['greenLED_smoothed','y'][time_ind,np.newaxis]))
                    redLED_L_data = np.hstack((interp_df['redLED_L_smoothed','x'][time_ind,np.newaxis],
                                            interp_df['redLED_L_smoothed','y'][time_ind,np.newaxis]))
                    redLED_C_data = np.hstack((interp_df['redLED_C_smoothed','x'][time_ind,np.newaxis],
                                            interp_df['redLED_C_smoothed','y'][time_ind,np.newaxis]))
                    redLED_R_data = np.hstack((interp_df['redLED_R_smoothed','x'][time_ind,np.newaxis],
                                            interp_df['redLED_R_smoothed','y'][time_ind,np.newaxis]))
                else:
                    greenLED_data = np.hstack((interp_df['greenLED','x'][time_ind,np.newaxis],
                                            interp_df['greenLED','y'][time_ind,np.newaxis]))
                    redLED_L_data = np.hstack((interp_df['redLED_L','x'][time_ind,np.newaxis],
                                            interp_df['redLED_L','y'][time_ind,np.newaxis]))
                    redLED_C_data = np.hstack((interp_df['redLED_C','x'][time_ind,np.newaxis],
                                            interp_df['redLED_C','y'][time_ind,np.newaxis]))
                    redLED_R_data = np.hstack((interp_df['redLED_R','x'][time_ind,np.newaxis],
                                            interp_df['redLED_R','y'][time_ind,np.newaxis]))
                if crop:
                    interp_centroid_data = np.hstack((centroid_dict['interp']['x'][time_ind,np.newaxis] - crop_offset_x,
                                                    centroid_dict['interp']['y'][time_ind,np.newaxis] - crop_offset_y))
                    dlc_centroid_data = np.hstack((centroid_dict['dlc']['x'][time_ind,np.newaxis] - crop_offset_x,
                                                centroid_dict['dlc']['y'][time_ind,np.newaxis] - crop_offset_y))
                    if smooth:
                        greenLED_data = np.hstack((interp_df['greenLED_smoothed','x'][time_ind,np.newaxis] - crop_offset_x,
                                                interp_df['greenLED_smoothed','y'][time_ind,np.newaxis] - crop_offset_y))
                        redLED_L_data = np.hstack((interp_df['redLED_L_smoothed','x'][time_ind,np.newaxis] - crop_offset_x,
                                                interp_df['redLED_L_smoothed','y'][time_ind,np.newaxis] - crop_offset_y))
                        redLED_C_data = np.hstack((interp_df['redLED_C_smoothed','x'][time_ind,np.newaxis] - crop_offset_x,
                                                interp_df['redLED_C_smoothed','y'][time_ind,np.newaxis] - crop_offset_y))
                        redLED_R_data = np.hstack((interp_df['redLED_R_smoothed','x'][time_ind,np.newaxis] - crop_offset_x,
                                                interp_df['redLED_R_smoothed','y'][time_ind,np.newaxis] - crop_offset_y))
                    else:
                        greenLED_data = np.hstack((interp_df['greenLED','x'][time_ind,np.newaxis] - crop_offset_x,
                                                interp_df['greenLED','y'][time_ind,np.newaxis] - crop_offset_y))
                        redLED_L_data = np.hstack((interp_df['redLED_L','x'][time_ind,np.newaxis] - crop_offset_x,
                                                interp_df['redLED_L','y'][time_ind,np.newaxis] - crop_offset_y))
                        redLED_C_data = np.hstack((interp_df['redLED_C','x'][time_ind,np.newaxis] - crop_offset_x,
                                                interp_df['redLED_C','y'][time_ind,np.newaxis] - crop_offset_y))
                        redLED_R_data = np.hstack((interp_df['redLED_R','x'][time_ind,np.newaxis] - crop_offset_x,
                                                interp_df['redLED_R','y'][time_ind,np.newaxis] - crop_offset_y))
                green_position_dot.set_offsets(greenLED_data)
                redL_position_dot.set_offsets(redLED_L_data)
                redC_position_dot.set_offsets(redLED_C_data)
                redR_position_dot.set_offsets(redLED_R_data)
                interp_position_dot.set_offsets(interp_centroid_data)
                assert time_ind == self.interp_df.index[time_ind]
                r = 30
                head_orient_line.set_data(
                    [interp_centroid_data[0], interp_centroid_data[0] +
                        r * np.cos(head_orientation[time_ind])],
                    [interp_centroid_data[1], interp_centroid_data[1] + r * np.sin(head_orientation[time_ind])],)
                time_delta = pd.Timedelta(self.interp_df['Datetime'].iloc[time_ind] - self.interp_df['Datetime'].iloc[0]).total_seconds()
                title.set_text(
                    f'time = {time_delta:3.4f}s\n frame = {self.interp_df.index[time_ind]}')
                
                redL_likelihood.set_data(
                    window_ind / sampling_frequency,
                    np.asarray(interp_df['redLED_L', 'likelihood'].iloc[time_ind + window_ind]))
                redR_likelihood.set_data(
                    window_ind / sampling_frequency,
                    np.asarray(interp_df['redLED_R', 'likelihood'].iloc[time_ind + window_ind]))
                green_likelihood.set_data(
                    window_ind / sampling_frequency,
                    np.asarray(interp_df['greenLED', 'likelihood'].iloc[time_ind + window_ind]))
                redC_likelihood.set_data(
                    window_ind / sampling_frequency,
                    np.asarray(interp_df['redLED_C', 'likelihood'].iloc[time_ind + window_ind]))
        #         redC_likelihood.set_data(
        #             window_ind / sampling_frequency,
        #             np.asarray(interp_df['redLED_C', 'likelihood'].iloc[time_ind + (window_size // 2) + window_ind]))
        #         green_likelihood.set_data(
        #             window_ind / sampling_frequency,
        #             np.asarray(interp_df['greenLED', 'likelihood'].iloc[time_ind + (window_size // 2) + window_ind]))
        #         redL_likelihood.set_data(
        #             window_ind / sampling_frequency,
        #             np.asarray(interp_df['redLED_L', 'likelihood'].iloc[time_ind + (window_size // 2) + window_ind]))
        #         redR_likelihood.set_data(
        #             window_ind / sampling_frequency,
        #             np.asarray(interp_df['redLED_R', 'likelihood'].iloc[time_ind + (window_size // 2) + window_ind]))
                progress_bar.update()

                return (image, interp_position_dot, dlc_position_dot, head_dir_GR_line, head_dir_BI_line, title, redC_likelihood, green_likelihood, redL_likelihood, redR_likelihood)

            movie = animation.FuncAnimation(fig, _update_plot, frames=n_frames,
                                        interval=1000 / fps, blit=True)
            if movie_name is not None: 
                movie.save(movie_name, writer=writer, dpi=400)
            video.release()

def convert_mp4(filename, video_path, dest_path, videotype, count_frames=False):
    """
    converts video to mp4 using passthrough for frames
    
    Parameters
    ----------
    filename : str
        filename of video including suffix (e.g. .h264)
    video_path : str
        path of input video excluding filename
    dest_path : str
        path of output video excluding filename, but no suffix (e.g. no '.mp4'
    videotype : str
        string of filetype to convert to (currently only accepts 'mp4')
    """
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

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

def diff(angle1, angle2):
    angle1 = np.rad2deg(angle1)
    angle2 = np.rad2deg(angle2)
    ang_diff = (angle2 - angle1 + 180) % 360 - 180
    where_below = np.where(ang_diff < -180)[0]
    ang_diff[where_below] = ang_diff[where_below] + 180
    return np.deg2rad(ang_diff)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def make_vector(x1, x2, y1=None, y2=None):
    pass
    
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
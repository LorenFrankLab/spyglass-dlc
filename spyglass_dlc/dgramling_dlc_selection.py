import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import datajoint as dj
from tqdm import tqdm as tqdm
import pynwb
import cv2
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_behav import RawPosition, VideoFile
from spyglass.common.common_interval import IntervalList
from spyglass_dlc.dgramling_dlc_pose_estimation import (
    DLCPoseEstimationSelection,
    DLCPoseEstimation,
)

from spyglass_dlc.dlc_utils import get_video_path
from .dgramling_dlc_centroid import DLCCentroid
from .dgramling_dlc_orient import DLCOrientation
from .dgramling_dlc_position import DLCSmoothInterp
from .dgramling_dlc_cohort import DLCSmoothInterpCohort

schema = dj.schema("dgramling_dlc_selection")


@schema
class DLCPosSelection(dj.Manual):
    """
    Specify collection of upstream DLCCentroid and DLCOrientation entries
    to combine into a set of position information
    """

    definition = """
    -> DLCCentroid.proj(dlc_si_cohort_centroid='dlc_si_cohort_selection_name', centroid_analysis_file_name='analysis_file_name')
    -> DLCOrientation.proj(dlc_si_cohort_orientation='dlc_si_cohort_selection_name', orientation_analysis_file_name='analysis_file_name')
    """


@schema
class DLCPos(dj.Computed):
    """
    Combines upstream DLCCentroid and DLCOrientation
    entries into a single entry with a single Analysis NWB file
    """

    definition = """
    -> DLCPosSelection
    ---
    -> AnalysisNwbfile
    position_object_id      : varchar(80)
    orientation_object_id   : varchar(80)
    velocity_object_id      : varchar(80)
    """

    def make(self, key):
        position_nwb_data = (DLCCentroid & key).fetch_nwb()[0]
        orientation_nwb_data = (DLCOrientation & key).fetch_nwb()[0]
        position_object = position_nwb_data["dlc_position"].spatial_series["position"]
        velocity_object = position_nwb_data["dlc_velocity"].time_series["velocity"]
        video_frame_object = position_nwb_data["dlc_velocity"].time_series[
            "video_frame_ind"
        ]
        orientation_object = orientation_nwb_data["dlc_orientation"].spatial_series[
            "orientation"
        ]
        position = pynwb.behavior.Position()
        orientation = pynwb.behavior.CompassDirection()
        velocity = pynwb.behavior.BehavioralTimeSeries()
        position.create_spatial_series(
            name=position_object.name,
            timestamps=np.asarray(position_object.timestamps),
            conversion=position_object.conversion,
            data=np.asarray(position_object.data),
            reference_frame=position_object.reference_frame,
            comments=position_object.comments,
            description=position_object.description,
        )
        orientation.create_spatial_series(
            name=orientation_object.name,
            timestamps=np.asarray(orientation_object.timestamps),
            conversion=orientation_object.conversion,
            data=np.asarray(orientation_object.data),
            reference_frame=orientation_object.reference_frame,
            comments=orientation_object.comments,
            description=orientation_object.description,
        )
        velocity.create_timeseries(
            name=velocity_object.name,
            timestamps=np.asarray(velocity_object.timestamps),
            conversion=velocity_object.conversion,
            unit=velocity_object.unit,
            data=np.asarray(velocity_object.data),
            comments=velocity_object.comments,
            description=velocity_object.description,
        )
        velocity.create_timeseries(
            name=video_frame_object.name,
            unit=video_frame_object.unit,
            timestamps=np.asarray(video_frame_object.timestamps),
            data=np.asarray(video_frame_object.data),
            description=video_frame_object.description,
            comments=video_frame_object.comments,
        )
        # Add to Analysis NWB file
        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
        nwb_analysis_file = AnalysisNwbfile()
        key["orientation_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], orientation
        )
        key["position_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], position
        )
        key["velocity_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], velocity
        )

        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )

        self.insert1(key)
        from .dgramling_position import PosSource

        key["source"] = "DLC"
        dlc_key = key.copy()
        valid_fields = PosSource().fetch().dtype.fields.keys()
        entries_to_delete = [entry for entry in key.keys() if entry not in valid_fields]
        for entry in entries_to_delete:
            del key[entry]
        PosSource().insert1(key=key, params=dlc_key, skip_duplicates=True)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        nwb_data = self.fetch_nwb()[0]
        index = pd.Index(
            np.asarray(nwb_data["position"].get_spatial_series().timestamps),
            name="time",
        )
        COLUMNS = [
            "video_frame_ind",
            "position_x",
            "position_y",
            "orientation",
            "velocity_x",
            "velocity_y",
            "speed",
        ]
        return pd.DataFrame(
            np.concatenate(
                (
                    np.asarray(
                        nwb_data["velocity"].time_series["video_frame_ind"].data,
                        dtype=int,
                    )[:, np.newaxis],
                    np.asarray(nwb_data["position"].get_spatial_series().data),
                    np.asarray(nwb_data["orientation"].get_spatial_series().data)[
                        :, np.newaxis
                    ],
                    np.asarray(nwb_data["velocity"].time_series["velocity"].data),
                ),
                axis=1,
            ),
            columns=COLUMNS,
            index=index,
        )


@schema
class DLCPosVideo(dj.Computed):
    """Creates a video of the computed head position and orientation as well as
    the original LED positions overlayed on the video of the animal.

    Use for debugging the effect of position extraction parameters."""

    definition = """
    -> DLCPos
    ---
    """

    def make(self, key):
        from tqdm import tqdm as tqdm

        M_TO_CM = 100
        epoch = (
            int(
                key["interval_list_name"]
                .replace("pos ", "")
                .replace(" valid times", "")
            )
            + 1
        )
        pose_estimation_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["interval_list_name"],
            "dlc_si_cohort_selection_name": key["dlc_si_cohort_centroid"],
        }
        pose_estimation_params, video_filename, output_dir = (
            DLCPoseEstimationSelection() & pose_estimation_key
        ).fetch1("pose_estimation_params", "video_path", "pose_estimation_output_dir")
        meters_per_pixel = (DLCPoseEstimation() & pose_estimation_key).fetch1(
            "meters_per_pixel"
        )
        crop = None
        if "cropping" in pose_estimation_params:
            crop = pose_estimation_params["cropping"]
        print("Loading position data...")
        position_info_df = (
            DLCPos()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
                "dlc_si_cohort_centroid": key["dlc_si_cohort_centroid"],
                "dlc_centroid_params_name": key["dlc_centroid_params_name"],
                "dlc_si_cohort_orientation": key["dlc_si_cohort_orientation"],
                "dlc_orientation_params_name": key["dlc_orientation_params_name"],
            }
        ).fetch1_dataframe()
        pose_estimation_df = pd.concat(
            {
                bodypart: (
                    DLCPoseEstimation.BodyPart()
                    & {**pose_estimation_key, **{"bodypart": bodypart}}
                ).fetch1_dataframe()
                for bodypart in (DLCSmoothInterpCohort.BodyPart & pose_estimation_key)
                .fetch("bodypart")
                .tolist()
            },
            axis=1,
        )
        assert len(pose_estimation_df) == len(position_info_df), (
            f"length of pose_estimation_df: {len(pose_estimation_df)} "
            f"does not match the length of position_info_df: {len(position_info_df)}."
        )

        nwb_base_filename = key["nwb_file_name"].replace(".nwb", "")
        if Path(output_dir).exists():
            output_video_filename = (
                f"{Path(output_dir).as_posix()}/"
                f"{nwb_base_filename}_{epoch:02d}_"
                f'{key["dlc_si_cohort_centroid"]}_'
                f'{key["dlc_centroid_params_name"]}'
                f'{key["dlc_orientation_params_name"]}.mp4'
            )
        else:
            output_video_filename = (
                f"{nwb_base_filename}_{epoch:02d}_"
                f'{key["dlc_si_cohort_centroid"]}_'
                f'{key["dlc_centroid_params_name"]}'
                f'{key["dlc_orientation_params_name"]}.mp4'
            )
        idx = pd.IndexSlice
        video_frame_inds = position_info_df["video_frame_ind"].astype(int).to_numpy()
        centroids = {
            bodypart: pose_estimation_df.loc[:, idx[bodypart, ("x", "y")]].to_numpy()
            for bodypart in pose_estimation_df.columns.levels[0]
        }
        position_mean = np.asarray(position_info_df[["position_x", "position_y"]])
        orientation_mean = np.asarray(position_info_df[["orientation"]])
        position_time = np.asarray(position_info_df.index)
        cm_per_pixel = meters_per_pixel * M_TO_CM

        print("Making video...")
        self.make_video(
            video_filename,
            centroids,
            position_mean,
            orientation_mean,
            position_time,
            video_frame_inds,
            output_video_filename=output_video_filename,
            cm_to_pixels=cm_per_pixel,
            disable_progressbar=False,
            crop=None,
        )

    @staticmethod
    def convert_to_pixels(data, frame_size, cm_to_pixels=1.0):
        """Converts from cm to pixels and flips the y-axis.
        Parameters
        ----------
        data : ndarray, shape (n_time, 2)
        frame_size : array_like, shape (2,)
        cm_to_pixels : float
        Returns
        -------
        converted_data : ndarray, shape (n_time, 2)
        """
        return data / cm_to_pixels

    @staticmethod
    def fill_nan(variable, video_time, variable_time):
        video_ind = np.digitize(variable_time, video_time[1:])

        n_video_time = len(video_time)
        try:
            n_variable_dims = variable.shape[1]
            filled_variable = np.full((n_video_time, n_variable_dims), np.nan)
        except IndexError:
            filled_variable = np.full((n_video_time,), np.nan)
        filled_variable[video_ind] = variable

        return filled_variable

    def make_video(
        self,
        video_filename,
        centroids,
        position_mean,
        orientation_mean,
        position_time,
        video_frame_inds,
        output_video_filename="output.mp4",
        cm_to_pixels=1.0,
        disable_progressbar=False,
        crop=None,
        arrow_radius=15,
        circle_radius=8,
    ):
        import matplotlib.animation as animation
        import matplotlib.font_manager as fm

        frame_offset = -1
        time_slice = []
        video_slowdown = 1
        vmax = 0.07  # ?
        # Set up formatting for the movie files

        window_size = 501

        window_ind = np.arange(window_size) - window_size // 2
        # Get video frames
        assert Path(
            video_filename
        ).exists(), f"Path to video: {video_filename} does not exist"
        color_swatch = [
            "#29ff3e",
            "#ff0073",
            "#ff291a",
            "#1e2cff",
            "#b045f3",
            "#ffe91a",
        ]
        video = cv2.VideoCapture(video_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = (int(video.get(3)), int(video.get(4)))
        frame_rate = video.get(5)
        Writer = animation.writers["ffmpeg"]
        fps = int(np.round(frame_rate / video_slowdown))
        writer = Writer(fps=fps, bitrate=-1)
        n_frames = np.max(video_frame_inds)
        ret, frame = video.read()
        print(f"initial frame: {video.get(1)}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if crop:
            frame = frame[crop[2] : crop[3], crop[0] : crop[1]].copy()
            crop_offset_x = crop[0]
            crop_offset_y = crop[2]
        frame_ind = 0
        ### TODO: ------ DELETE THIS------- #####
        n_frames = int(n_frames // 15)
        with plt.style.context("dark_background"):
            # Set up plots
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [8, 1]},
                constrained_layout=False,
            )

            axes[0].tick_params(colors="white", which="both")
            axes[0].spines["bottom"].set_color("white")
            axes[0].spines["left"].set_color("white")
            image = axes[0].imshow(frame, animated=True)
            print(f"frame after init plot: {video.get(1)}")
            centroid_plot_objs = {
                bodypart: axes[0].scatter(
                    [],
                    [],
                    s=2,
                    zorder=102,
                    color=color,
                    label=f"{bodypart} position",
                    animated=True,
                    alpha=0.6,
                )
                for color, bodypart in zip(color_swatch, centroids.keys())
            }
            centroid_position_dot = axes[0].scatter(
                [],
                [],
                s=5,
                zorder=102,
                color="#b045f3",
                label="centroid position",
                animated=True,
                alpha=0.6,
            )
            (orientation_line,) = axes[0].plot(
                [],
                [],
                color="cyan",
                linewidth=1,
                animated=True,
                label="Orientation",
            )
            axes[0].set_xlabel("")
            axes[0].set_ylabel("")
            ratio = frame_size[1] / frame_size[0]
            if crop:
                ratio = (crop[3] - crop[2]) / (crop[1] - crop[0])
            x_left, x_right = axes[0].get_xlim()
            y_low, y_high = axes[0].get_ylim()
            axes[0].set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
            #     axes[0].set_xlim(position_info[position_name[0]].min() - 10,
            #                      position_info[position_name[0]].max() + 10)
            #     axes[0].set_ylim(position_info[position_name[1]].min() - 10,
            #                      position_info[position_name[1]].max() + 10)
            axes[0].spines["top"].set_color("black")
            axes[0].spines["right"].set_color("black")
            time_delta = pd.Timedelta(
                position_time[0] - position_time[0]
            ).total_seconds()
            axes[0].legend(loc="lower right", fontsize=4)
            title = axes[0].set_title(
                f"time = {time_delta:3.4f}s\n frame = {frame_ind}",
                fontsize=8,
            )
            fontprops = fm.FontProperties(size=12)
            #     scalebar = AnchoredSizeBar(axes[0].transData,
            #                                20, '20 cm', 'lower right',
            #                                pad=0.1,
            #                                color='white',
            #                                frameon=False,
            #                                size_vertical=1,
            #                                fontproperties=fontprops)

            #     axes[0].add_artist(scalebar)
            axes[0].axis("off")
            # if plot_likelihood:
            #     (redL_likelihood,) = axes[1].plot(
            #         [],
            #         [],
            #         color="yellow",
            #         linewidth=1,
            #         animated=True,
            #         clip_on=False,
            #         label="red_left",
            #     )
            #     (redR_likelihood,) = axes[1].plot(
            #         [],
            #         [],
            #         color="blue",
            #         linewidth=1,
            #         animated=True,
            #         clip_on=False,
            #         label="red_right",
            #     )
            #     (green_likelihood,) = axes[1].plot(
            #         [],
            #         [],
            #         color="green",
            #         linewidth=1,
            #         animated=True,
            #         clip_on=False,
            #         label="green",
            #     )
            #     (redC_likelihood,) = axes[1].plot(
            #         [],
            #         [],
            #         color="red",
            #         linewidth=1,
            #         animated=True,
            #         clip_on=False,
            #         label="red_center",
            #     )
            #     axes[1].set_ylim((0.0, 1))
            #     axes[1].set_xlim(
            #         (
            #             window_ind[0] / sampling_frequency,
            #             window_ind[-1] / sampling_frequency,
            #         )
            #     )
            #     axes[1].set_xlabel("Time [s]")
            #     axes[1].set_ylabel("Likelihood")
            #     axes[1].set_facecolor("black")
            #     axes[1].spines["top"].set_color("black")
            #     axes[1].spines["right"].set_color("black")
            #     axes[1].legend(loc="upper right", fontsize=4)
            progress_bar = tqdm(leave=True, position=0)
            progress_bar.reset(total=n_frames)

            def _update_plot(time_ind):
                if time_ind == 0:
                    video.set(1, time_ind + 1)
                ret, frame = video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if crop:
                        frame = frame[crop[2] : crop[3], crop[0] : crop[1]].copy()
                    image.set_array(frame)
                pos_ind = np.where(video_frame_inds == time_ind)[0]
                if len(pos_ind) == 0:
                    centroid_position_dot.set_offsets((np.NaN, np.NaN))
                    for bodypart in centroid_plot_objs.keys():
                        centroid_plot_objs[bodypart].set_offsets((np.NaN, np.NaN))
                    orientation_line.set_data((np.NaN, np.NaN))
                    title.set_text(f"time = {0:3.4f}s\n frame = {time_ind}")
                else:
                    pos_ind = pos_ind[0]
                    dlc_centroid_data = self.convert_to_pixels(
                        position_mean[pos_ind], frame, cm_to_pixels
                    )
                    if crop:
                        dlc_centroid_data = np.hstack(
                            (
                                self.convert_to_pixels(
                                    position_mean[pos_ind, 0, np.newaxis],
                                    frame,
                                    cm_to_pixels,
                                )
                                - crop_offset_x,
                                self.convert_to_pixels(
                                    position_mean[pos_ind, 1, np.newaxis],
                                    frame,
                                    cm_to_pixels,
                                )
                                - crop_offset_y,
                            )
                        )
                    for bodypart in centroid_plot_objs.keys():
                        centroid_plot_objs[bodypart].set_offsets(
                            self.convert_to_pixels(
                                centroids[bodypart][pos_ind], frame, cm_to_pixels
                            )
                        )
                    centroid_position_dot.set_offsets(dlc_centroid_data)
                    r = 30
                    orientation_line.set_data(
                        [
                            dlc_centroid_data[0],
                            dlc_centroid_data[0]
                            + r * np.cos(orientation_mean[pos_ind]),
                        ],
                        [
                            dlc_centroid_data[1],
                            dlc_centroid_data[1]
                            + r * np.sin(orientation_mean[pos_ind]),
                        ],
                    )
                    # Need to convert times to datetime object probably.

                    time_delta = pd.Timedelta(
                        pd.to_datetime(position_time[pos_ind] * 1e9, unit="ns")
                        - pd.to_datetime(position_time[0] * 1e9, unit="ns")
                    ).total_seconds()
                    title.set_text(f"time = {time_delta:3.4f}s\n frame = {time_ind}")
                # redL_likelihood.set_data(
                #     window_ind / sampling_frequency,
                #     np.asarray(
                #         interp_df["redLED_L", "likelihood"].iloc[time_ind + window_ind]
                #     ),
                # )
                # redR_likelihood.set_data(
                #     window_ind / sampling_frequency,
                #     np.asarray(
                #         interp_df["redLED_R", "likelihood"].iloc[time_ind + window_ind]
                #     ),
                # )
                # green_likelihood.set_data(
                #     window_ind / sampling_frequency,
                #     np.asarray(
                #         interp_df["greenLED", "likelihood"].iloc[time_ind + window_ind]
                #     ),
                # )
                # redC_likelihood.set_data(
                #     window_ind / sampling_frequency,
                #     np.asarray(
                #         interp_df["redLED_C", "likelihood"].iloc[time_ind + window_ind]
                #     ),
                # )
                progress_bar.update()

                return (
                    image,
                    centroid_position_dot,
                    orientation_line,
                    title,
                    # redC_likelihood,
                    # green_likelihood,
                    # redL_likelihood,
                    # redR_likelihood,
                )

            movie = animation.FuncAnimation(
                fig,
                _update_plot,
                frames=np.arange(0, n_frames + 3),
                interval=1000 / fps,
                blit=True,
            )
            movie.save(output_video_filename, writer=writer, dpi=400)
            video.release()

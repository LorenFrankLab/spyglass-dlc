import bottleneck
import numpy as np
import pandas as pd
import datajoint as dj
import pynwb
import pynwb.behavior
from position_tools import (
    get_angle,
    get_centriod,
    get_distance,
    get_speed,
    get_velocity,
    interpolate_nan,
)
from position_tools.core import gaussian_smooth
from spyglass.common.dj_helper_fn import fetch_nwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_behav import RawPosition

schema = dj.schema("dgramling_trodes_position")


@schema
class TrodesPosParams(dj.Manual):
    """
    Parameters for calculating the position (centroid, velocity, orientation)
    """

    definition = """
    trodes_pos_params_name: varchar(80) # name for this set of parameters
    ---
    params: longblob
    """

    @classmethod
    def insert_default(cls, **kwargs):
        """
        Insert default parameter set for position determination
        """
        params = {
            "max_separation": 9.0,
            "max_speed": 300.0,
            "position_smoothing_duration": 0.125,
            "speed_smoothing_std_dev": 0.100,
            "orient_smoothing_std_dev": 0.001,
            "led1_is_front": 1,
            "is_upsampled": 0,
            "upsampling_sampling_rate": None,
            "upsampling_interpolation_method": "linear",
        }
        cls.insert1(
            {"trodes_pos_params_name": "default", "params": params},
            skip_duplicates=True,
        )


@schema
class TrodesPosSelection(dj.Manual):
    """
    Table to pair an interval with position data
    and position determination parameters
    """

    definition = """
    -> RawPosition
    -> TrodesPosParams
    ---
    """


@schema
class TrodesPos(dj.Computed):
    """
    Table to calculate the position based on Trodes tracking
    """

    definition = """
    -> TrodesPosSelection
    ---
    -> AnalysisNwbfile
    position_object_id : varchar(80)
    orientation_object_id : varchar(80)
    velocity_object_id : varchar(80)
    """

    def make(self, key):
        print(f"Computing position for: {key}")
        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
        raw_position = (RawPosition() & key).fetch_nwb()[0]
        position_info_parameters = (TrodesPosParams() & key).fetch1("params")
        position = pynwb.behavior.Position()
        orientation = pynwb.behavior.CompassDirection()
        velocity = pynwb.behavior.BehavioralTimeSeries()

        METERS_PER_CM = 0.01

        try:
            # calculate the processed position
            spatial_series = raw_position["raw_position"]
            position_info = self.calculate_position_info_from_spatial_series(
                spatial_series,
                position_info_parameters["max_separation"],
                position_info_parameters["max_speed"],
                position_info_parameters["speed_smoothing_std_dev"],
                position_info_parameters["position_smoothing_duration"],
                position_info_parameters["orient_smoothing_std_dev"],
                position_info_parameters["led1_is_front"],
                position_info_parameters["is_upsampled"],
                position_info_parameters["upsampling_sampling_rate"],
                position_info_parameters["upsampling_interpolation_method"],
            )
            # create nwb objects for insertion into analysis nwb file
            position.create_spatial_series(
                name="position",
                timestamps=position_info["time"],
                conversion=METERS_PER_CM,
                data=position_info["position"],
                reference_frame=spatial_series.reference_frame,
                comments=spatial_series.comments,
                description="x_position, y_position",
            )

            orientation.create_spatial_series(
                name="orientation",
                timestamps=position_info["time"],
                conversion=1.0,
                data=position_info["orientation"],
                reference_frame=spatial_series.reference_frame,
                comments=spatial_series.comments,
                description="orientation",
            )

            velocity.create_timeseries(
                name="velocity",
                timestamps=position_info["time"],
                conversion=METERS_PER_CM,
                unit="m/s",
                data=np.concatenate(
                    (position_info["velocity"], position_info["speed"][:, np.newaxis]),
                    axis=1,
                ),
                comments=spatial_series.comments,
                description="x_velocity, y_velocity, speed",
            )
            velocity.create_timeseries(
                name="video_frame_ind",
                unit="index",
                timestamps=position_info["time"],
                data=raw_position["raw_position"].data,
                description="video_frame_ind",
                comments=spatial_series.comments,
            )
        except ValueError:
            pass

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()

        key["position_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], position
        )
        key["orientation_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], orientation
        )
        key["velocity_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], velocity
        )

        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])

        self.insert1(key)

    def calculate_position_info_from_spatial_series(
        self,
        spatial_series,
        max_LED_separation,
        max_plausible_speed,
        speed_smoothing_std_dev,
        position_smoothing_duration,
        orient_smoothing_std_dev,
        led1_is_front,
        is_upsampled,
        upsampling_sampling_rate,
        upsampling_interpolation_method,
    ):

        CM_TO_METERS = 100

        # Get spatial series properties
        time = np.asarray(spatial_series.timestamps)  # seconds
        position = np.asarray(spatial_series.data)  # meters

        # remove NaN times
        is_nan_time = np.isnan(time)
        position = position[~is_nan_time]
        time = time[~is_nan_time]

        dt = np.median(np.diff(time))
        sampling_rate = 1 / dt
        meters_to_pixels = spatial_series.conversion

        # Define LEDs
        if led1_is_front:
            front_LED = position[:, [0, 1]].astype(float)
            back_LED = position[:, [2, 3]].astype(float)
        else:
            back_LED = position[:, [0, 1]].astype(float)
            front_LED = position[:, [2, 3]].astype(float)

        # Convert to cm
        back_LED *= meters_to_pixels * CM_TO_METERS
        front_LED *= meters_to_pixels * CM_TO_METERS

        # Set points to NaN where the front and back LEDs are too separated
        dist_between_LEDs = get_distance(back_LED, front_LED)
        is_too_separated = dist_between_LEDs >= max_LED_separation

        back_LED[is_too_separated] = np.nan
        front_LED[is_too_separated] = np.nan

        # Calculate speed
        front_LED_speed = get_speed(
            front_LED,
            time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )
        back_LED_speed = get_speed(
            back_LED,
            time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )

        # Set to points to NaN where the speed is too fast
        is_too_fast = (front_LED_speed > max_plausible_speed) | (
            back_LED_speed > max_plausible_speed
        )
        back_LED[is_too_fast] = np.nan
        front_LED[is_too_fast] = np.nan

        # Interpolate the NaN points
        back_LED = interpolate_nan(back_LED)
        front_LED = interpolate_nan(front_LED)

        # Smooth
        moving_average_window = int(position_smoothing_duration * sampling_rate)
        back_LED = bottleneck.move_mean(
            back_LED, window=moving_average_window, axis=0, min_count=1
        )
        front_LED = bottleneck.move_mean(
            front_LED, window=moving_average_window, axis=0, min_count=1
        )

        if is_upsampled:
            position_df = pd.DataFrame(
                {
                    "time": time,
                    "back_LED_x": back_LED[:, 0],
                    "back_LED_y": back_LED[:, 1],
                    "front_LED_x": front_LED[:, 0],
                    "front_LED_y": front_LED[:, 1],
                }
            ).set_index("time")

            upsampling_start_time = time[0]
            upsampling_end_time = time[-1]

            n_samples = (
                int(
                    np.ceil(
                        (upsampling_end_time - upsampling_start_time)
                        * upsampling_sampling_rate
                    )
                )
                + 1
            )
            new_time = np.linspace(
                upsampling_start_time, upsampling_end_time, n_samples
            )
            new_index = pd.Index(
                np.unique(np.concatenate((position_df.index, new_time))), name="time"
            )
            position_df = (
                position_df.reindex(index=new_index)
                .interpolate(method=upsampling_interpolation_method)
                .reindex(index=new_time)
            )

            time = np.asarray(position_df.index)
            back_LED = np.asarray(position_df.loc[:, ["back_LED_x", "back_LED_y"]])
            front_LED = np.asarray(position_df.loc[:, ["front_LED_x", "front_LED_y"]])

            sampling_rate = upsampling_sampling_rate

        # Calculate position, orientation, velocity, speed
        position = get_centriod(back_LED, front_LED)  # cm

        orientation = get_angle(back_LED, front_LED)  # radians
        is_nan = np.isnan(orientation)

        # Unwrap orientation before smoothing
        orientation[~is_nan] = np.unwrap(orientation[~is_nan])
        orientation[~is_nan] = gaussian_smooth(
            orientation[~is_nan],
            orient_smoothing_std_dev,
            sampling_rate,
            axis=0,
            truncate=8,
        )
        # convert back to between -pi and pi
        orientation[~is_nan] = np.angle(np.exp(1j * orientation[~is_nan]))

        velocity = get_velocity(
            position,
            time=time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )  # cm/s
        speed = np.sqrt(np.sum(velocity**2, axis=1))  # cm/s

        return {
            "time": time,
            "position": position,
            "orientation": orientation,
            "velocity": velocity,
            "speed": speed,
        }

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

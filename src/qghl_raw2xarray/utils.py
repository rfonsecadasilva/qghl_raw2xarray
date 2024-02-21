"""
Module with utility functions for handling time, paddle,
station, and coordinate attributes in a dataset.
"""
import numpy as np


def set_time_attrs(time="UTC +10:00"):
    """
    Set the attributes for the time variable.

    Parameters:
    time (str): The time zone offset in the format "UTC +/-HH:MM".
    Default is "UTC +10:00".

    Returns:
    dict: A dictionary containing the attributes for the time variable.
    """
    if time[:3] == "UTC":
        return {"standard_name": 'time', "long_name": f"Time ({time})", "axis": "time"}
    return {"standard_name": 'time', "long_name": f"Time ({time})",
             "units": "s", "axis": "time"}


def time_offset(ds, start_time=None):
    """
    Calculate the time offset of each time value relative to the start time of the experiment.

    Parameters:
    ds (xarray.Dataset): The dataset containing the time variable.
    start_time (numpy.datetime64, optional): The start time
    of the experiment. 
    If not provided, the first time value in the dataset will be used.

    Returns:
    ds (xarray.Dataset): The dataset with the added 'time_offset' variable.

    """
    if start_time is None:
        start_time = ds.time[0].values
        if isinstance(start_time, np.datetime64):
            ds = ds.assign_coords(
                {"time_offset": ("time", (ds.time.values-start_time).astype(float)/1e9)})
        else:
            # assuming in seconds
            ds = ds.assign_coords(
                {"time_offset": ("time", (ds.time.values-start_time).astype(float))})
        ds.time_offset.attrs = set_time_attrs(
            "Relative to first time value of experiment")
    else:
        ds = ds.assign_coords(
            {"time_offset": ("time", (ds.time.values-start_time).astype(float)/1e9)})
        ds.time_offset.attrs = set_time_attrs(
            "Relative to start time of experiment")
    return ds


def set_paddle_coords(ds, **kwargs):
    """
    Set paddle coordinates and attributes in a dataset.

    Parameters:
    ds (xarray.Dataset): The dataset to modify.
    **kwargs: Keyword arguments containing the paddle coordinates and attributes.

    Returns:
    xarray.Dataset: The modified dataset with updated paddle coordinates and attributes.
    """
    for paddle_arg in ["paddle", "paddle_x", "paddle_y"]:
        if paddle_arg in kwargs:
            ds = ds.assign_coords(
                {paddle_arg: ("paddle", kwargs[paddle_arg])})
            if f"{paddle_arg}_attrs" in kwargs:
                ds[paddle_arg].attrs = kwargs[f"{paddle_arg}_attrs"]
    return ds


def set_station_coords(ds, **kwargs):
    """
    Set station coordinates and attributes in a dataset.

    Parameters:
    ds (xarray.Dataset): The dataset to modify.
    **kwargs: Keyword arguments containing the station coordinates and attributes.

    Returns:
    xarray.Dataset: The modified dataset with updated station coordinates and attributes.
    """
    for station_arg in ["station", "station_x", "station_y", "station_z",
                        "station_x_pres", "station_y_pres", "station_z_pres",
                        "station_zbed", "station_ID", "station_h"]:
        if station_arg in kwargs:
            ds = ds.assign_coords(
                {station_arg: ("station", kwargs[station_arg])})
            if f"{station_arg}_attrs" in kwargs:
                ds[station_arg].attrs = kwargs[f"{station_arg}_attrs"]
    return ds


def set_time_coords(ds, start_time=None, **kwargs):
    """
    Set time coordinates and attributes in a dataset.

    Parameters:
    ds (xarray.Dataset): The dataset to modify.
    start_time (numpy.datetime64, optional): The start time of the experiment.
    **kwargs: Keyword arguments containing the time coordinates and attributes.

    Returns:
    xarray.Dataset: The modified dataset with updated time coordinates and attributes.
    """
    # set time attributes
    if "time_attrs" in kwargs:
        ds.time.attrs = kwargs["time_attrs"]
    # set time_utc coordinates
    if "time_utc_diff" in kwargs:
        ds = ds.assign_coords(
            {"time_utc": (ds.time + np.timedelta64(kwargs["time_utc_diff"], 'h'))})
        ds["time_utc"].attrs = set_time_attrs(time="UTC")
    # set time offset coordinate relative to start time
    ds = time_offset(ds, start_time)
    return ds

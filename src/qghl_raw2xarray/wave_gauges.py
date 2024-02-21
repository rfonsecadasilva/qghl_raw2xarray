"""
This module provides functions for reading and processing wave gauge data.
"""
import datetime
import mikeio
import numpy as np
import pandas as pd
import xarray as xr
from qghl_raw2xarray.utils import set_station_coords, set_time_coords


def calculate_ds_cwg(cwgfile, cwgpath='./', tmin=None, start_time=None, wg_dt=0.01, **kwargs):
    """
    Calculate an xarray Dataset from a capacitance wave gauge file.

    Parameters:
    cwgfile (str): The name of the capacitance wave gauge file.
    cwgpath (str, optional): The path to the capacitance wave gauge file. Defaults to './'.
    tmin (numpy.datetime64, optional): The start time of time series. Defaults to None.
    start_time (numpy.datetime64, optional): The start time of the experiment. Defaults to None.
    wg_dt (float, optional): The time step of the wave gauge data. Defaults to 0.01.
    kwargs (dict): Additional keyword arguments for setting coordinates and attributes.

    Returns:
    ds (xarray.Dataset): The xarray Dataset containing the wave gauge data.

    Raises:
    FileNotFoundError: If the specified file is not found.

    """
    print("Reading capacitance wave gauge file", cwgfile)
    try:
        cwgts = pd.read_csv(cwgpath+cwgfile, header=None,
                            index_col=None, usecols=range(8))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"The specified file {cwgpath+cwgfile} was not found.") from e
    cwgts.columns = [f"C{i+1:1d}" for i in cwgts.columns]
    # create data_vars
    data_vars = {"Watlev": (("station", "time"), cwgts.values.transpose(),
                            {"standard_name": "Watlev",
                            "long_name": "Water level",
                             "units": "m"})}
    # time axis in seconds from start of experiment
    time_axis = wg_dt * np.arange(0, len(cwgts.iloc[:, 0]))
    if tmin is not None:
        # convert time axis to datetime64 from tmin
        time_axis = np.array(
            [tmin + np.timedelta64(int(s * 1e9), 'ns') for s in time_axis])
    coords = {"time": time_axis}
    # create xarray
    ds = xr.Dataset(data_vars=data_vars,
                    coords=coords)
    # set station coordinates and attributes
    ds = set_station_coords(ds, **kwargs)
    # # set time coordinates and attributes
    ds = set_time_coords(ds, start_time, **kwargs)
    # set run attributes
    if "attrs_run" in kwargs:
        ds.attrs = kwargs["attrs_run"]
    # set wave gauge comment
    if "wg_comment" in kwargs:
        ds.attrs["Gauge position"] = kwargs["wg_comment"]
    ds.attrs["Description"] = "HR Wallingford capacitance wave gauge data"
    ds.attrs["Raw files path"] = cwgpath
    ds.attrs["Raw files"] = [cwgfile]
    ds.attrs["Xarray dataset date"] = str(
        np.datetime64(datetime.datetime.now().isoformat()))
    return ds


def calculate_ds_rwg(rwgfile, rwgpath='./', tmin=None, start_time=None, **kwargs):
    """
    Calculate xarray dataset from resistance wave gauge data.

    Parameters:
    rwgfile (str): The name of the resistance wave gauge file.
    rawpath (str, optional): The path to the resistance wave gauge file. Defaults to './'.
    tmin (numpy.datetime64, optional): The start time of time series. Defaults to None.
    start_time (numpy.datetime64, optional): The start time of the experiment. Defaults to None.
    **kwargs: Additional keyword arguments for setting coordinates and attributes.

    Returns:
    xr.Dataset: The xarray dataset containing the resistance wave gauge data.

    Raises:
    FileNotFoundError: If the specified file is not found.
    """
    print("Reading resistance wave gauge file", rwgfile)
    try:
        rwgts = mikeio.read(rwgpath+rwgfile)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"The specified file {rwgpath+rwgfile} was not found.") from e
    rwg_data = np.array(
        [rwgts[i].values for i in rwgts.to_xarray().data_vars if i != "CH11 Volt Trig"])
    data_vars = {"Watlev": (("station", "time"), rwg_data,
                            {"standard_name": "Watlev",
                            "long_name": "Water level",
                             "units": "m"})}
    time_axis = rwgts.time.values
    if tmin is None:
        # calculate tmin from XML file
        tmin = calculate_tmin_rwgxml(rwgfile, rwgpath)
    # convert time axis to datetime64 from tmin
    time_axis = tmin + (time_axis - time_axis[0])
    coords = {"time": time_axis}
    # create xarray
    ds = xr.Dataset(data_vars=data_vars,
                    coords=coords)
    # set station coordinates and attributes
    ds = set_station_coords(ds, **kwargs)
    # # set time coordinates and attributes
    ds = set_time_coords(ds, start_time, **kwargs)
    # set run attributes
    if "attrs_run" in kwargs:
        ds.attrs = kwargs["attrs_run"]
    # set wave gauge comment
    if "wg_comment" in kwargs:
        ds.attrs["Gauge position"] = kwargs["wg_comment"]
    ds.attrs["Description"] = "DHI resistance wave gauge data"
    ds.attrs["Raw files path"] = rwgpath
    ds.attrs["Raw files"] = [rwgfile]
    ds.attrs["Xarray dataset date"] = str(
        np.datetime64(datetime.datetime.now().isoformat()))
    return ds

def calculate_tmin_rwgxml(rwgfile, rwgpath):
    """
    Calculate the start time of the resistance wave gauge data from the XML file.

    Parameters:
    rwgfile (str): The name of the resistance wave gauge file.
    rwgpath (str): The path to the resistance wave gauge file.

    Returns:
    numpy.datetime64: The start time of the resistance wave gauge data.

    Raises:
    FileNotFoundError: If the specified file is not found.
    """
    try:
        xml_file = open(f"{rwgpath}{rwgfile[:-5]}.xml", "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"The specified file {rwgpath+rwgfile[:-5]}.xml was not found.") from e
    rwg_tmax_str = xml_file.split("<date>")[1].split("</date>")[0]
    rwg_tmax = np.datetime64(rwg_tmax_str)
    rwg_tmax += np.timedelta64(int(rwg_tmax_str[-5:-3]), 'h') + \
        np.timedelta64(int(rwg_tmax_str[-2:]), 'm')
    rwg_dur = float(xml_file.split("<DurationSeconds>")[1]
                    .split("</DurationSeconds>")[0])
    return rwg_tmax + np.timedelta64(int(-rwg_dur * 1e9), 'ns')
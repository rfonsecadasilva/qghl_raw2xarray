"""
This module provides functions for reading and processing Flowtracker ADV and 
pressure sensor data.
"""
import datetime
import json
import numpy as np
import xarray as xr
from qghl_raw2xarray.utils import set_station_coords, set_time_coords


def calculate_ds_ft(ftfile, ftpath, start_time=None, **kwargs):
    """
    Calculate the Xarray dataset for Flowtracker ADV and pressure sensor data.

    Parameters:
    ftfile (str): The name of the Flowtracker csv file.
    ftpath (str): Path to the directory containing the Flowtracker data file.
    start_time (numpy.datetime64, optional): The start time of the experiment. Defaults to None.
    **kwargs: Additional keyword arguments.

    Returns:
    xr.Dataset: Xarray dataset containing the Flowtracker data.
    """
    ds = ftcsv2ds(ftfile, ftpath)
    # expand dimensions to include station
    ds = ds.expand_dims({"station": ["FT"]})
    # set station coordinates and attributes
    ds = set_station_coords(ds, **kwargs)
    # # set time coordinates and attributes
    ds = set_time_coords(ds, start_time, **kwargs)
    # set run attributes
    if "attrs_run" in kwargs:
        ds.attrs.update(kwargs["attrs_run"])
    # set adv orientation attributes
    if "adv_orientation" in kwargs:
        ds.attrs["ADV_orientation"] = kwargs["adv_orientation"]
    ds.attrs["Description"] = "Flowtracker ADV and pressure sensor data"
    ds.attrs["Raw files path"] = ftpath
    ds.attrs["Raw files"] = [ftfile]
    if "ft_serialnumber" in kwargs:
        ds.attrs["FT Serial Number"] = kwargs["ft_serialnumber"]
    ds.attrs["Xarray dataset date"] = str(
        np.datetime64(datetime.datetime.now().isoformat()))
    return ds


def ftcsv2ds(ftfile, ftpath):
    """
    Convert Flowtracker2 csv file into xarray dataset.

    Parameters:
    ftfile (str): The name of the Flowtracker csv file.
    ftpath (str): Path to the directory containing the Flowtracker data file.

    Raises:
    FileNotFoundError: If the specified FT csv or config file is not found.     

    Returns:
    ds (xarray.Dataset): Xarray dataset containing the Flowtracker data.   
    """
    def str_to_nptime(x):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    try:
        csvfile = open(f"{ftpath}{ftfile}.csv", "r",
                       encoding="utf-8").readlines()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"The specified file {ftpath}{ftfile}.csv was not found.") from e
    print("Reading Flowtracker csv file", ftfile)
    try:
        cfgfile = open(f"{ftpath}{ftfile}.labadv_config",
                       "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"The specified file {ftpath}{ftfile}.labadv_config was not found.") from e
    nbeams = len(
        [i for i in csvfile[0].strip().split(",") if "Correlation" in i])
    utc_time = np.array([str_to_nptime(i.strip().split(",")[1])
                        for i in csvfile[1:]])
    time = np.array([str_to_nptime(i.strip().split(",")[2])
                    for i in csvfile[1:]])
    ds = []  # initialise list of datasets
    # part 1 - velocities, corrected velocities, correlation, SNR, amplitude, noise
    ds.append(ft_vel_reader(csvfile, nbeams))
    # part 2 - temperature, sound speed, realtime pressure,
    # corrected pressure, depth, ps calibration, voltage
    ds.append(ft_unidim_reader(csvfile, nbeams))
    # part 3 - accelerometer
    ds.append(ft_accel_reader(csvfile, nbeams))
    ds = xr.merge(ds)
    ds = ds.assign_coords({"beam": np.arange(1, nbeams+1),
                          "time": time})
    ds = ds.assign_coords({"utc_time": ("time", utc_time)})
    ds.attrs = json.loads(cfgfile.replace("\n", "")).items()
    # edit attributes to be able to save to netcdf
    # convert None to string "None"
    ds.attrs = {k: v if v is not None else str(v) for k, v in ds.attrs.items()}
    ds.attrs = {k: v if not isinstance(
        v, bool) else 1 if v else 0 for k, v in ds.attrs.items()}  # convert boolean to 1 or 0
    # remove "/" from attribute name to be able to save it to netcdf
    ds.attrs['SoundSpeedOverride (m.s)'] = ds.attrs.pop(
        'SoundSpeedOverride (m/s)')
    ds["beam"].attrs = {"standard_name": "beam", "units": "", "Direction": [
        "Beam1|X", "Beam2|Y", "Beam3|Z"]}  # assign beam metadata
    return ds


def ft_vel_reader(csvfile, nbeams=3):
    """
    Read Flowtracker velocities and associated beam-dependent parameters from a csv file
    and returns it as an xarray Dataset.

    Parameters:
    csvfile (list): List of strings representing the contents of the csv file.
    nbeams (int, optional): Number of beams used by the Flowtracker. Default is 3.

    Returns:
    xr.Dataset: A dataset containing the Flowtracker velocities, corrected velocities, 
    correlation, SNR, amplitude and noise, with dimensions 'beam' and 'time'.

    """
    csv_var_beam = [i.split(".")[0] for i in csvfile[0].strip().split(",")[
        3:3+nbeams*6:nbeams]]
    csv_var_beam_units = [i.split(".")[1].split()[-1][1:-1]
                          for i in csvfile[0].strip().split(",")[3:3+nbeams*6:nbeams]]
    data_vars = {
        l: (["beam", "time"],
            np.transpose(
            [[
                float(d) if d else np.nan
                for d in j.strip().split(",")[3+idx*nbeams:3+idx*nbeams+nbeams]
            ]
                for j in csvfile[1:]]
        ),
            {"standard_name": l, "units": csv_var_beam_units[idx]})
        for idx, l in enumerate(csv_var_beam)
    }
    return xr.Dataset(
        data_vars=data_vars,
    )


def ft_unidim_reader(csvfile, nbeams=3):
    """
    Read Flowtracker undidimensional parameters from a csv file and returns 
    it as an xarray Dataset.

    Parameters:
    csvfile (list): List of strings representing the contents of the csv file.
    nbeams (int, optional): Number of beams. Defaults to 3.

    Returns:
    xr.Dataset: A dataset containing the temperature, sound speed, realtime 
    ressure data, corrected pressure, depth, calibration age, and voltage,
    with dimension 'time'.
    """
    # corrected pressure, depth, ps calibration, voltage
    csv_var = [i.split("(")[0][:-1]
               for i in csvfile[0].strip().split(",")[3+nbeams*6:3+nbeams*6+7]]
    csv_var_units = [i.split(
        "(")[1][:-1] for i in csvfile[0].strip().split(",")[3+nbeams*6:3+nbeams*6+7]]
    data_vars = {
        l: (
            ["time"],
            np.array(
                [[
                    float(d) if d else np.nan
                    for d in j.strip().split(",")[3+nbeams*6+idx:3+nbeams*6+idx+1]
                ]
                    for j in csvfile[1:]
                ]
            ).squeeze(),
            {"standard_name": l, "units": csv_var_units[idx]}
        )
        for idx, l in enumerate(csv_var)
    }
    return xr.Dataset(
        data_vars=data_vars,
    )


def ft_accel_reader(csvfile, nbeams=3):
    """
    Reads the accelerometer data from a CSV file and returns it as an xarray Dataset.
    Accelerometer was assumed to have beam dimension (rather than x, y, z dimensions)
    for simplicity.

    Parameters:
    csvfile (list): List of strings representing the CSV file contents.
    nbeams (int, optional): Number of beams in the flow tracker system (default is 3).

    Returns:
    xr.Dataset: An xarray Dataset containing the accelerometer data, with dimensions 
    'beam' and 'time'.
    """
    csv_var_beam_2 = [i.split(".")[0] for i in csvfile[0].strip().split(",")[
        3+nbeams*6+7:3+nbeams*6+7+nbeams*1:nbeams]]
    # assumption as csv file does not have accelerometer units
    csv_var_beam_2_units = ['m/s^2']
    data_vars = {
        l: (
            ["beam", "time"],
            np.transpose(
                [[float(d) for d in j.strip().split(",")[3+nbeams*6+7:3+nbeams*6+7+nbeams]]
                 for j in csvfile[1:]]
            ),
            {"standard_name": l, "units": csv_var_beam_2_units[idx]
             }
        )
        for idx, l in enumerate(csv_var_beam_2)
    }
    return xr.Dataset(
        data_vars=data_vars,
    )

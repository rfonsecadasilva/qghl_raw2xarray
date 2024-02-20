"""
This module provides functions for reading and processing Vectrino point ADV data.
"""
import datetime
import os
import re
import dolfyn
import numpy as np
import xarray as xr
from qghl_raw2xarray.utils import set_station_coords, set_time_coords


def calculate_ds_vec(vecfile, vecpath, tmin=None, start_time=None, **kwargs):
    """
    Calculate the xarray dataset from Vectrino point ADV data.

    Parameters:
    vecfile (str): Name of Vectrino data file.
    vecpath (str): Path to the directory containing
    the Vectrino data file.
    tmin (numpy.datetime64, optional): The start time of time series. Defaults to None.
    start_time (numpy.datetime64, optional): The start time of the experiment. Defaults to None.
    **kwargs: Additional keyword arguments to be passed to the set_station_coords and 
    set_time_coords functions.

    Returns:
    ds (xarray.Dataset): Xarray dataset containing the Vectrino point ADV data.
    """
    if vecfile.split(".")[-1] != "vno":
        # # create xarray through parsing of hdr, dat, and pck files
        ds = vec2ds(vecfile=vecfile, vecpath=vecpath)
    else:
        # create xarray with dolfyn package
        ds = vec2ds_dolfyn(vecfile=vecfile, vecpath=vecpath)
    if tmin is not None:
        ds['time'] = ds['time'] - ds['time'][0] + tmin
    # expand dimensions to include station
    ds = ds.expand_dims({"station": ["VEC"]})
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
    ds.attrs["Description"] = "Vectrino point ADV data"
    ds.attrs["Raw files path"] = vecpath
    ds.attrs["Raw files"] = [vecfile]
    ds.attrs["Xarray dataset date"] = str(
        np.datetime64(datetime.datetime.now().isoformat()))
    return ds


def vec2ds_dolfyn(vecfile, vecpath):
    """
    Convert Vectrino vno file into xarray dataset.

    Parameters:
    vecfile (str): Vectrino vno filename.
    vecpath (str): Vectrino file directory.

    Returns:
    ds (xarray.Dataset): Xarray dataset containing the Vectrino data.
    """
    print(f"Reading Vectrino vno file {vecfile}")
    ds = dolfyn.read(f"{vecpath}{vecfile}")
    return ds


def vec2ds(vecfile, vecpath):
    """
    Convert Vectrino hdr and datfiles into xarray dataset.

    Parameters:
    vecfile (str): Vectrino filename (without extension).
    This holds both for hdr and dat files.
    vecpath (str): Vectrino file directory.

    Raises:
    FileNotFoundError: If the specified hdr file is not found.

    Returns:
    ds (xarray.Dataset): Xarray dataset containing the Vectrino data.
    """
    def str_to_nptime(x):
        return datetime.datetime.strptime(x, "%d/%m/%Y %I:%M:%S %p")
    ds = []  # initialize list with xarray datasets
    hdrfilename = f'{vecpath}{vecfile}.hdr'
    print(f"Reading Vectrino file {hdrfilename}")
    try:
        hdrfile = open(hdrfilename, "r", encoding="utf-8").readlines()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"The specified file {hdrfilename} was not found.") from e
    nbeams = [int(re.split(r'\s{2,}', j.strip())[-1])
              for j in hdrfile if "Number of beams" in j][0]
    tini_str = [re.split(r'\s{2,}', j.strip())[-1]
                for j in hdrfile if "Time of first measurement" in j][0]
    sampling_rate = [re.split(r'\s{2,}', j.strip())[-1].split()[0]
                     for j in hdrfile if "Sampling rate" in j][0]
    dt = datetime.timedelta(seconds=1/float(sampling_rate))
    # pck file
    pck_file = f'{vecpath}{vecfile}.pck'
    if pck_file in os.getcwd():
        ds.append(vec_pck_reader(pck_file, vecpath, hdrfile, nbeams))
    # dat file
    dat_file = [j for j in os.listdir(f'{vecpath}') if
                hdrfilename.split("/")[-1].split(".")[0] in j
                and j.split(".")[-1] in ["dat"]
                ][0]
    dat_file = f'{vecpath}{dat_file}'
    ds.append(vec_dat_reader(dat_file, hdrfile, nbeams))
    ds = xr.merge(ds)
    time = np.arange(str_to_nptime(tini_str), str_to_nptime(
        tini_str)+dt*(len(ds["time"])), dt)
    ds = ds.assign_coords({"beam": np.arange(1, nbeams+1),
                           "time": time})
    ds.attrs = {
        re.split(
            r'\s{2,}', j.strip()
        )[0].replace("/", " "):
        re.split(r'\s{2,}', j.strip())[1]
        if len(re.split(r'\s{2,}', j.strip(
        ))) > 1 else
        "" for j in hdrfile
        if j.strip() != "" and j[0].isalpha()}  # assign overall metadata (attrs cannot have '/')
    ds["beam"].attrs = {"standard_name": "beam", "units": "", "Direction": [
        "Beam1|X", "Beam2|Y", "Beam3|Z", "Beam4|Z2"]}  # assign beam metadata
    return ds


def vec_pck_reader(pck_file, vecpath, hdrfile, nbeams):
    """
    Read and process a pck file from a Vectrino instrument.

    Parameters:
    pck_file (str): The name of the pck file to be read.
    vecpath (str): The path to the directory containing the pck file.
    hdrfile (list): A list of strings representing the header file.
    nbeams (int): The number of beams in the Vectrino instrument.

    Returns:
    ds (xarray.Dataset): A dataset containing the processed data from the pck file.

    """
    print("read pck file")
    pckf = open(f'{vecpath}{pck_file}', "r", encoding="utf-8").readlines()
    counter = np.arange(
        1, len(open(f'{vecpath}{pck_file}', "r", encoding="utf-8").readlines())//2+1)
    # line with info on data variables and units
    idx = [k+1 for k, j in enumerate(hdrfile) if pck_file in j][0]
    pck_var = [re.split(r'\s{2,}', hdrfile[j].strip())[1].split()[
        0] for j in range(idx+2, idx+2+nbeams*1, nbeams)]
    pck_units = [hdrfile[j].strip().split()[-1][1:-1]
                 for j in range(idx+2, idx+2+nbeams*1, nbeams)]
    # cycle 1 and 2 (assuming there are two cycles in the file)
    cycle = [1, 2]
    data_vars = {
        "pck_"+l: (
            ["cycle", "beam", "counter"], np.transpose(
                [[float(d) for d in j.strip().split()[2+idx*nbeams:2+idx*nbeams+nbeams]]
                    for j in pckf]
            ).reshape(len(cycle), nbeams, len(counter)),
            {"standard_name": l, "units": pck_units[idx]}
        )
        for idx, l in enumerate(pck_var)
    }
    ds = xr.Dataset(
        data_vars=data_vars,
    )
    ds = ds.assign_coords(
        {"pck_Sample": ("counter", np.array(
            [int(i.strip().split()[0])
                for i in pckf[:len(counter)]]),
            {"standard_name": "Sample", "units": ""}
        )
        }
    )
    ds = ds.assign_coords(
        {"pck_Distance": ("counter", np.array(
            [float(i.strip().split()[1])
                for i in pckf[:len(counter)]]),
            {"standard_name": "Distance", "units": "mm"})
         }
    )
    return ds


def vec_dat_reader(datfile, hdrfile, nbeams):
    """
    Read and process a dat file from a Vectrino instrument.

    Parameters:
    datfile (str): The name of the dat file to be read.
    hdrfile (list): A list of strings representing the header file.
    nbeams (int): The number of beams in the Vectrino instrument.

    Returns:
    ds (xarray.Dataset): A dataset containing the processed data from the dat file.

    """
    print("read dat file")
    datf = open(datfile, "r", encoding="utf-8").readlines()
    # line with info on data variables and units
    idx = [k+1 for k, j in enumerate(hdrfile)
           if datfile.split("/")[-1] in j][0]
    dat_var = [
        re.split(
            r'\s{2,}', hdrfile[j].strip()
        )[1].split()[0]
        for j in range(idx+4, idx+4+nbeams*4, nbeams)
    ]
    # ignoring four first rows (e.g., status and ensemble counter)
    #
    dat_units = [
        hdrfile[j].strip().split()[-1][1:-1]
        for j in range(
            idx+4, idx+4+nbeams*4, nbeams)
    ]  # 2 because file mark and time are absent
    data_vars = {
        l: (
            ["beam", "time"], np.transpose(
                [[float(d) for d in j.strip().split()[2+idx*nbeams:2+idx*nbeams+nbeams]]
                 for j in datf]
            ),
            {"standard_name": l, "units": dat_units[idx]})
        for idx, l in enumerate(dat_var)
    }
    ds = xr.Dataset(
        data_vars=data_vars,
    )
    ds = ds.assign_coords(
        {"Ensemble_counter": (
            "time", np.array(
                [int(i.strip().split()[0])
                 for i in datf]),
            {"standard_name": "Ensemble", "units": ""}
        )
        }
    )
    ds = ds.assign_coords(
        {"Status": ("time", np.array(
            [int(i.strip().split()[1])
             for i in datf]),
            {"standard_name": "Status", "units": ""})
         }
    )
    return ds

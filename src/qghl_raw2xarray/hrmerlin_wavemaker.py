"""
This module provides functions for reading and processing HR Merling wave maker data.
"""
import datetime
import os
import struct
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
from qghl_raw2xarray.utils import set_paddle_coords, set_time_coords


def calculate_ds_wm(xmlfile, xmlfilepath, tmin=None, start_time=None, **kwargs):
    """
    Reads HR Merlin wave maker binary file and returns xarray dataset.

    Parameters:
    xmlfile (str): The name of the HR Merlin xml file.
    xmlfilepath (str): The path to the directory containing the xml file and binary files.
    tmin (numpy.datetime64, optional): The start time of time series. Defaults to None.
    start_time (numpy.datetime64, optional): The start time of the experiment. Defaults to None.
    **kwargs: Additional keyword arguments for setting attributes and coordinates.

    Returns:
    xr.Dataset: The xarray dataset containing the wave maker data.

    Raises:
    FileNotFoundError: If the xml file or binary files are not found.

    """
    # parse HR Merlin xml file
    try:
        xml_tree = ET.parse(xmlfilepath+xmlfile)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"The specified file {xmlfilepath+xmlfile} was not found."
            ) from e
    xml_root = xml_tree.getroot()
    # extract metadata from XML
    pnts = int(get_xml_value(xml_root, "Points"))
    chans = int(get_xml_value(xml_root, "Channels"))
    rate = float(get_xml_value(xml_root, "Rate"))  # assumed to be in Hz
    # list of variables and units
    list_channels = [get_xml_value(xml_root, tag) for tag in [
        f"Channel{i+1}" for i in range(chans)]]
    list_variables = [i.split("(")[0][:-1] for i in list_channels]
    list_units = [i.split("(")[1].split(")")[0] for i in list_channels]
    # list of bin files
    list_filename = [f for f in os.listdir(xmlfilepath) if f.startswith(xmlfile.split(
        ".")[0]) and f.endswith(".bin")]  # list with bin files, one for each paddle
    data = []  # initialize list with numpy array, one for each paddle
    for filename in list_filename:
        print(f"Reading HR Merlin wave maker file {filename}")
        data.append(read_binary_file(filename, xmlfilepath, pnts, chans))
    data = np.concatenate([np.expand_dims(i, axis=0) for i in data], axis=0)
    # create xarray data_vars and coords
    data_vars = {list_variables[d]: (("paddle", "time"), data[:, d, :],
                                     {"standard_name": list_variables[d],
                                     "units": list_units[d]}) for d in range(chans)}
    # time axis in seconds from start of experiment
    time_axis = np.arange(0, pnts/rate, 1/rate)
    if tmin is not None:
        # convert time axis to datetime64 from tmin
        time_axis = np.array(
            [tmin + np.timedelta64(int(i * 1e9), 'ns') for i in time_axis])
    coords = {"time": time_axis}
    # create xarray dataset
    ds = xr.Dataset(data_vars=data_vars,
                    coords=coords)
    # set paddle coordinates and attributes
    ds = set_paddle_coords(ds, **kwargs)
    # # set time coordinates and attributes
    ds = set_time_coords(ds, start_time, **kwargs)
    # set run attributes
    if "attrs_run" in kwargs:
        ds.attrs = kwargs["attrs_run"]
    ds.attrs["Description"] = "HR Merlin wave gauge data at wave maker"
    ds.attrs["Raw files path"] = xmlfilepath
    ds.attrs["Raw files"] = list_filename
    ds.attrs["Xarray dataset date"] = str(
        np.datetime64(datetime.datetime.now().isoformat()))
    return ds


def read_binary_file(filename, xmlfilepath, pnts, chans):
    """
    Read binary file and return unpacked data.

    Parameters:
    filename (str): The name of the binary file.
    xmlfilepath (str): The path to the XML file.
    pnts (int): The number of points.
    chans (int): The number of channels.

    Returns:
    np.ndarray: The unpacked data from the binary file.
    """
    with open(xmlfilepath + filename, 'rb') as file:
        binary_data = file.read()
    data_format = '<' + str(len(binary_data) // 4) + \
        'f'  # Little-endian, float32
    unpacked_data = np.array(struct.unpack(data_format, binary_data))
    if pnts * chans * 4 == len(binary_data):
        data = unpacked_data.reshape((chans, pnts))
    else:
        unpacked_data = np.concatenate(
            [unpacked_data, np.full(pnts * chans - len(unpacked_data), np.nan)])
        data = np.array(unpacked_data).reshape((chans, pnts))
    return data


def get_xml_value(xml_root, tag):
    """
    Helper function to get the value of a tag from XML.

    Parameters:
    xml_root (Element): The root element of the XML.
    tag (str): The tag to search for in the XML.

    Returns:
    str or None: The value of the tag if found, None otherwise.
    """
    element = xml_root.find(".//" + tag)
    return element.text if element is not None else None

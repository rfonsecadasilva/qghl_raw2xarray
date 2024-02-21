# qghl_raw2xarray
 
 This project contains functions to read QGHL reef project raw files and transform then into xarray datasets.
 
 It contains functions to read and process:
 - HR Merlin wave maker binary files
 - HR Walling capacitance wave gauge csv files
 - DHI resistance wave gauge dfs0 (DHI-Mike) files
 - Nortek Vectrino point ADV binary vno files
 - Xylem Flowtracker ADV and pressure sensor csv files

## To install this package, execute the following commands:
```
python -m pip install git+https://github.com/rfonsecadasilva/qghl_raw2xarray
```
## Alternatively for developer mode:
```
git clone https://github.com/rfonsecadasilva/qghl_raw2xarray.git
cd qghl_raw2xarray
pip install -e .
```

### Example of usage - to create a xarray dataset from HR Merlin binary wave maker files
```
from qghl_raw2xarray import hrmerlin_wavemaker as hrm
ds_wm=hrm.calculate_ds_wm(xmlfile='T001.xml')
```

### Example of usage - to create a xarray dataset from HR Wallingford capacitance csv wave gauge files
```
from qghl_raw2xarray import wave_gauges as wg
ds_cwg=wg.calculate_ds_cwg(cwgfile='T001_cwg.csv')
```

### Example of usage - to create a xarray dataset from DHI resistance dfs0 wave gauge files
```
from qghl_raw2xarray import wave_gauges as wg
ds_rwg=wg.calculate_ds_rwg(rwgfile='T001.dfs0')
```

### Example of usage - to create a xarray dataset from Vectrino ADV binary vno files
```
from qghl_raw2xarray import vectrino as vec
vec.calculate_ds_vec(vecfile='T001.vno')
```

### Example of usage - to create a xarray dataset from Flowtracker ADV and pressure sensor csv files 
```
from qghl_raw2xarray import flowtracker as ft
ft.calculate_ds_ft(ftfile='T001_ft.csv')
```

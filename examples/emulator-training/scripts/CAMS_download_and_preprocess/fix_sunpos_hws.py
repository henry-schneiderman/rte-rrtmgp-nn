"""
Created on Wed Sep 15 17:00:33 2021

@author: peter

Modified by Henry Schneiderman Oct. 12, 2023
"""


from netCDF4 import Dataset,num2date
import numpy as np
from sunposition import sunpos
# Note pysolar requires conda environment 'pytorch2.0_tmp"
# pysolar is under the GPL license - so don't use!
#from pysolar.solar import get_altitude
import datetime
import pytz


# Input data file
fpath = "/data-T1/hws/CAMS/original_data/n2o/tmp.7.nc"

dat         = Dataset(fpath,'a')

p   = dat.variables['pressure'][:,:,:].data  # pres_level
lon = np.rad2deg(dat.variables['clon'][:].data)
lat = np.rad2deg(dat.variables['clat'][:].data)
timedat = dat.variables['time'][:]

ntime = p.shape[0]
nsite = p.shape[2]
nlev = p.shape[1]


# save solar zenith angle

t_unit =  dat.variables['time'].units + '-00:00'
t_cal  =  dat.variables['time'].calendar

lonn = lon.reshape(1,nsite).repeat(ntime,axis=0)
latt = lat.reshape(1,nsite).repeat(ntime,axis=0)
lonn = lonn.reshape(nsite*ntime); 
latt = latt.reshape(nsite*ntime)

timedatt = dat.variables['time'][:].reshape(ntime,1).repeat(nsite,axis=1)
timedatt = timedatt.reshape(nsite*ntime)

t_unit =  dat.variables['time'].units 


times = num2date(timedatt,units = t_unit,calendar = t_cal).data


if True:
    az,zen = sunpos(times.data,latt,lonn,0)[:2] #discard RA, dec, H

    sza_new = zen.reshape(ntime,nsite)
    sza = dat.createVariable("solar_zenith_angle","f4",("time", "cell"))
    sza[:] = sza_new[:]
else:
    timezone = pytz.timezone('UTC') #datetime.timezone.utc

    times = num2date(timedatt,units = t_unit,calendar = t_cal, 
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True).data


    times = [timezone.localize(t) for t in times]

    sza_new_2 = np.array([90.0 - get_altitude(latt[i],lonn[i],t) for i, t in enumerate(times)])
    sza_new_2 = np.reshape(sza_new_2,(ntime,nsite))
    
    sza_2 = dat.createVariable("solar_zenith_angle_2","f4",("time", "cell"))
    sza_2[:,:] = sza_new_2[:,:]

dat.close()


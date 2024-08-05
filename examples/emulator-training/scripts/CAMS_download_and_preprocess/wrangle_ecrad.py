# run with: conda activate netcdf3.8
from netCDF4 import Dataset,num2date
import numpy as np
import xarray as xr
import os
import sys

def generate_raw_sources(totplanck, min_temp, temp_layers):
    #double totplnk(bnd, temperature_Planck)
    # examples, layers, bands
    raw_sources = torch.zeros(temp_layers.shape[0],temp_layers.shape[1], totplanck.shape[0])

    for i in torch.arange(temp_layers.shape[0]):
        for j in torch.arange(temp_layers.shape[1]):
            diff = temp_layers[i,j] - min_temp
            index = np.floor(diff)
            fraction = diff - np.float(index)
            raw_sources[i,j,:] = totplanck[:,index](1.0 - fraction) + totplanck[:,index + 1] * fraction

    return raw_sources

def wrapper_raw_sources (mode,month,year, base_directory, planck_file_name="../../../rte-rrtmgp-nn/rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"):

    dt = Dataset(planck_file_name,'r')

    totplanck = dt.variables["totplnk"][:,:].data
    temp_planck = dt.variables["temperature_Planck"][:].data
    min_temp = temp_planck[0]
    dt.close()

    temp_file_name = f'{base_directory}lw_input-{mode}-{year}-{month}.nc'

    ######

    dt = Dataset(temp_file_name,'r')
    temp_layer = dt.variables["temp_layer"][:,:,:].data
    shape = temp_layer.shape

    col = shape[0]*shape[1]

    temp_layer = temp_layer.reshape((col, shape[2]))

    temp_skin = dt.variable["skin_temperature"][:].data
    temp_skin = temp_skin.reshape((-1,1))

    temp_layer = torch.stack((temp_layer, temp_skin), axis=1)

    raw_sources = generate_raw_sources(totplanck, min_temp, temp_layer)

    dt.close()

    ###########

    source_file_name = f'{base_directory}lw_source-{mode}-{year}-{month}.nc'
    dt.Dataset(source_file_name, "w")

    dim1 = dt.createDimension("col",col)
    dim2 = dt.createDimension("level",temp_layer.shape[1])
    dim3 = dt.createDimension("band",totplanck.shape[0])

    var = dt.createVariable("raw_sources","f4",("col","level","band"))
    var[:]=raw_sources[:]
    dt.close()

    
def wrangle_ecrad_data(mode,month,year, base_directory):
    d = base_directory + f'{mode}/{year}/{month}/'

    f = d + f'CAMS_{year}-{month}.final.2.nc'
    o1 = d + "tmp.1.nc"
    o2 = d + "tmp.2.nc"

    ref = '/home/hws/ecrad/practical/era5slice.nc'

    cmd = f'ncrename -d level,half_level {f} {o1}'
    os.system(cmd)

    cmd = f'ncrename -v level,half_level {o1} {o2}'
    os.system(cmd)
    cmd = f'rm -f {o1}'
    os.system(cmd)

    cmd = f'ncrename -d layer,level {o2} {o1}'
    os.system(cmd)
    cmd = f'rm -f {o2}'
    os.system(cmd)

    cmd = f'ncrename -v layer,level {o1} {o2}'
    os.system(cmd)
    cmd = f'rm -f {o1}'
    os.system(cmd)

    # It is an incorrect constant
    cmd = f'ncks -x -v surface_emissivity {o2} {o1}'
    os.system(cmd)
    cmd = f'rm -f {o2}'
    os.system(cmd)

    # It is an incorrect constant
    cmd = f'ncks -x -v cloud_fraction  {o1} {o2}'
    os.system(cmd)
    cmd = f'rm -f {o1}'
    os.system(cmd)

    # constant for entire atmosphere
    cmd = f'ncks -A -v o2_vmr {ref} {o2}'
    os.system(cmd)

    dt = Dataset(o2,'a')

    sites = dt.variables['site'][:].data
    n_sites = sites.shape[0]

    t = dt.variables['time'][:].data
    n_time = t.shape[0]

    n_col = n_time * n_sites

    half_level = dt.variables['half_level'][:].data
    n_half_level = half_level.shape[0]

    print(f'n_site = {n_sites}')
    print(f'n_time = {n_time}')
    print(f'n_half_level = {n_half_level}')
    n_level = n_half_level - 1

    col = dt.createDimension("col", n_col)

    new_data = np.full(n_col, 0.95, dtype=np.float32)
    lw_emissivity = dt.createVariable("lw_emissivity","f4",("col",))
    lw_emissivity[:]=new_data[:]
    lw_emissivity.setncattr("units","1")
    lw_emissivity.setncattr("long_name","Longwave surface emissivity")

    albedo_data = dt.variables['surface_albedo'][:,:].data
    albedo_data =np.reshape(albedo_data, (n_col,))

    sw_albedo = dt.createVariable("sw_albedo","f4",("col",))
    sw_albedo[:]=albedo_data[:]
    sw_albedo.setncattr("units","1")
    sw_albedo.setncattr("long_name","Shortwave surface albedo")

    data = dt.variables['surface_temperature'][:,:].data
    data =np.reshape(data, (n_col,))

    var = dt.createVariable("skin_temperature","f4",("col",))
    var[:]=data[:]
    var.setncattr("units","K")
    var.setncattr("long_name","Skin temperature")

    #####

    data = dt.variables['solar_zenith_angle'][:,:].data
    data =np.reshape(data, (n_col,))
    data = np.cos(data * np.pi / 180.0)

    var2 = dt.createVariable("cos_solar_zenith_angle","f4",("col",))
    var2[:]=data[:]
    var2.setncattr("units","1")
    var2.setncattr("long_name","Cosine of the solar zenith angle")

    ####

    data = dt.variables['pres_level'][:,:,:].data
    data = np.reshape(data, (n_col,n_half_level))

    var3 = dt.createVariable("pressure_hl","f4",("col","half_level"))
    var3[:]=data[:]
    var3.setncattr("units","Pa")
    var3.setncattr("long_name","Pressure at half-levels")

    ####

    data = dt.variables['temp_level'][:,:,:].data
    data = np.reshape(data, (n_col,n_half_level))

    var4 = dt.createVariable("temperature_hl","f4",("col","half_level"))
    var4[:]=data[:]
    var4.setncattr("units","K")
    var4.setncattr("long_name","Temperature at half-levels")

    ####

    data = dt.variables['water_vapor'][:,:,:].data
    data =np.reshape(data, (n_col,n_level))

    # input is vmr
    m_dry = 28.964
    m_h2o =  18.01528

    # converting from vmr to mass ratio
    data = data * m_h2o / m_dry

    # converting from mass ratio to specific humidity
    data = data / (1.0 + data)

    var5 = dt.createVariable("q","f4",("col","level"))
    var5[:]=data[:]
    var5.setncattr("units","1")
    var5.setncattr("long_name","Specific humidity")

    m_o3 = 47.99820

    data = dt.variables['ozone'][:,:,:].data
    data =np.reshape(data, (n_col,n_level))

    # converting from vmr to mass ratio
    data = data * m_o3 / m_dry

    var6 = dt.createVariable("o3_mmr","f4",("col","level"))
    var6[:]=data[:]
    var6.setncattr("units","1")
    var6.setncattr("long_name","Ozone mass mixing ratio")

    #####
    new_data = np.full((n_col, n_level), 0.1, dtype=np.float32)
    var7 = dt.createVariable("cloud_fraction","f4",("col","level"))
    var7[:]=new_data[:]
    var7.setncattr("units","1")
    var7.setncattr("long_name","Cloud fraction")

    data = dt.variables['clwc'][:,:,:].data
    data = np.reshape(data, (n_col,n_level))
    # converting from specific content to mass ratio

    data = data / (1.0 - data)

    var8 = dt.createVariable("q_liquid","f4",("col","level"))
    var8[:]=data[:]
    var8.setncattr("units","1")
    var8.setncattr("long_name","Gridbox-mean liquid mixing ratio")

    data = dt.variables['ciwc'][:,:,:].data
    data = np.reshape(data, (n_col,n_level))
    # converting from specific content to mass ratio

    data = data / (1.0 - data)

    var9 = dt.createVariable("q_ice","f4",("col","level"))
    var9[:]=data[:]
    var9.setncattr("units","1")
    var9.setncattr("long_name","Gridbox-mean ice mixing ratio")

    new_data = np.full((n_col, n_level), 25.0e-6, dtype=np.float32)

    var10 = dt.createVariable("re_ice","f4",("col","level"))
    var10[:]=new_data[:]
    var10.setncattr("units","m")
    var10.setncattr("long_name","Ice effective radius")

    new_data = np.full((n_col, n_level), 14.0e-6, dtype=np.float32)

    var11 = dt.createVariable("re_liquid","f4",("col","level"))
    var11[:]=new_data[:]
    var11.setncattr("units","m")
    var11.setncattr("long_name","Liquid effective radius")

    data = dt.variables['carbon_dioxide'][:,:,:].data
    data = np.reshape(data, (n_col,n_level))
    data = data * 1.0e-06

    var12 = dt.createVariable("co2_vmr","f4",("col","level"))
    var12[:]=data[:]
    var12.setncattr("units","1")
    var12.setncattr("long_name","CO2 volume mixing ratio")


    data = dt.variables['methane'][:,:,:].data
    data = np.reshape(data, (n_col,n_level))
    data = data * 1.0e-09

    var13 = dt.createVariable("ch4_vmr","f4",("col","level"))
    var13[:]=data[:]
    var13.setncattr("units","1")
    var13.setncattr("long_name","CH4 volume mixing ratio")

    data = dt.variables['nitrous_oxide'][:,:,:].data
    data = np.reshape(data, (n_col,n_level))
    data = data * 1.0e-09

    var14 = dt.createVariable("n2o_vmr","f4",("col","level"))
    var14[:]=data[:]
    var14.setncattr("units","1")
    var14.setncattr("long_name","N2O volume mixing ratio")

    data = dt.variables['carbon_monoxide'][:,:,:].data
    data = np.reshape(data, (n_col,n_level))


    var15 = dt.createVariable("co_vmr","f4",("col","level"))
    var15[:]=data[:]
    var15.setncattr("units","1")
    var15.setncattr("long_name","CO volume mixing ratio")

    data = dt.variables['nitrogen_dioxide'][:,:,:].data
    data = np.reshape(data, (n_col,n_level))
    data = data * 1.0e-06

    var16 = dt.createVariable("no2_vmr","f4",("col","level"))
    var16[:]=data[:]
    var16.setncattr("units","1")
    var16.setncattr("long_name","NO2 volume mixing ratio")

    dt.close()


    cmd = f'ncks -x -v surface_albedo {o2} {o1}'
    os.system(cmd)
    cmd = f'rm -f {o2}'
    os.system(cmd)

    cmd = f'ncks -x -v surface_temperature {o1} {o2}'
    os.system(cmd)
    cmd = f'rm -f {o1}'
    os.system(cmd)

    cmd = f'ncks -x -v solar_zenith_angle {o2} {o1}'
    os.system(cmd)
    cmd = f'rm -f {o2}'
    os.system(cmd)

    cmd = f'ncks -x -v pres_level {o1} {o2}'
    os.system(cmd)
    cmd = f'rm -f {o1}'
    os.system(cmd)

    cmd = f'ncks -x -v temp_level {o2} {o1}'
    os.system(cmd)
    cmd = f'rm -f {o2}'
    os.system(cmd)

    cmd = f'ncks -x -v water_vapor {o1} {o2}'
    os.system(cmd)
    cmd = f'rm -f {o1}'
    os.system(cmd)

    cmd = f'ncks -x -v ozone {o2} {o1}'
    os.system(cmd)
    cmd = f'rm -f {o2}'
    os.system(cmd)

    cmd = f'ncks -x -v clwc {o1} {o2}'
    os.system(cmd)
    cmd = f'rm -f {o1}'
    os.system(cmd)

    cmd = f'ncks -x -v ciwc {o2} {o1}'
    os.system(cmd)
    cmd = f'rm -f {o2}'
    os.system(cmd)

    cmd = f'ncks -x -v carbon_dioxide {o1} {o2}'
    os.system(cmd)
    cmd = f'rm -f {o1}'
    os.system(cmd)

    cmd = f'ncks -x -v methane {o2} {o1}'
    os.system(cmd)
    cmd = f'rm -f {o2}'
    os.system(cmd)

    cmd = f'ncks -x -v nitrous_oxide {o1} {o2}'
    os.system(cmd)
    cmd = f'rm -f {o1}'
    os.system(cmd)

    cmd = f'ncks -x -v carbon_monoxide {o2} {o1}'
    os.system(cmd)
    cmd = f'rm -f {o2}'
    os.system(cmd)

    output_name = d + f'lw_input-{mode}-{year}-{month}.nc'

    cmd = f'ncks -x -v nitrogen_dioxide {o1} {output_name}'
    os.system(cmd)
    cmd = f'rm -f {o1}'
    os.system(cmd)
    # organize for processing


if __name__ == "__main__":

    #mode = 'testing'
    #month = '03'
    #year = '2009'

    base_directory = f'/data-T1/hws/CAMS/processed_data/'

    months = [str(m).zfill(2) for m in range(1,13)]

    combo = [('training','2008'),('cross_validation','2008'),('testing','2009'),('testing','2015'),('testing','2020')]

    for c in combo:
        mode = c[0]
        year = c[1]
        print(f'Processing {mode} {year}')
        for month in months[:]:
            print(f'{month}')
            wrangle_ecrad_data(mode, month, year, base_directory)




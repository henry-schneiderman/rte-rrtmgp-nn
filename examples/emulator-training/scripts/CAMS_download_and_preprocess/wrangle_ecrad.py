# run with: conda activate netcdf3.8
from netCDF4 import Dataset,num2date
import numpy as np
import xarray as xr
import os
import sys

def generate_raw_sources(totplanck, min_temp, max_temp, temp_layers, bandwidth):
    #double totplnk(bnd, temperature_Planck)
    # examples, layers, bands
    raw_sources = np.zeros((temp_layers.shape[0],temp_layers.shape[1], totplanck.shape[0]), dtype=np.float32)

    if True:
        print(f"Min allowable temp = {min_temp}. Actual min temp = {np.min(temp_layers)}")
        print(f"Max allowable temp = {max_temp}. Actual max temp = {np.max(temp_layers)}")

    if np.min(temp_layers) < min_temp:
        print(f"Out of range: Min allowable temp = {min_temp}. Actual min temp = {np.min(temp_layers)}")
        exit()

    if np.max(temp_layers) > max_temp:
        print(f"Out of range: Max allowable temp = {max_temp}. Actual max temp = {np.max(temp_layers)}")
        exit()

    for i in np.arange(temp_layers.shape[0]):
        for j in np.arange(temp_layers.shape[1]):
            diff = temp_layers[i,j] - min_temp
            index = np.int32(np.floor(diff))
            fraction = diff - np.float32(index)
            raw_sources[i,j,:] = 1.0e04 * bandwidth[:] * np.pi * (totplanck[:,index] * (1.0 - fraction) + totplanck[:,index + 1] * fraction)

    return raw_sources



def wrapper_raw_sources (mode,month,year, base_directory, planck_file_name="/home/hws/rrtmg_lw.nc"): #planck_file_name="../../../../rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"):

    dt = Dataset(planck_file_name,'r')

    #totplanck = dt.variables["totplnk"][:,:].data
    totplanck = dt.variables["IntegratedPlanckFunction"][:,:].data
    #temp_planck = dt.variables["temperature_Planck"][:].data
    temp_planck = dt.variables["TemperaturePlanckValues"][:].data

    #NOTE Adding 1 to min temp to agree with radiation_ifs_rrtm.F90 in ecRad!!!
    min_temp = temp_planck[0] + 1  
    max_temp = temp_planck[-1]

    bandlowerlimit = dt.variables["BandWavenumberLowerLimit"][:].data
    bandupperlimit = dt.variables["BandWavenumberUpperLimit"][:].data
    bandwidth = bandupperlimit - bandlowerlimit

    dt.close()

    temp_file_name = f'{base_directory}{mode}/{year}/{month}/lw_input-{mode}-{year}-{month}.nc'

    ######

    dt = Dataset(temp_file_name,'r')
    temp_layer = dt.variables["temp_layer"][:,:,:].data
    shape = temp_layer.shape

    col = shape[0]*shape[1]

    temp_layer = temp_layer.reshape((col, shape[2]))

    temp_skin = dt.variables["skin_temperature"][:].data
    temp_skin = temp_skin.reshape((-1,1))

    temp_layer = np.concatenate((temp_layer, temp_skin), axis=1)

    raw_sources = generate_raw_sources(totplanck, min_temp, max_temp, temp_layer, bandwidth)

    dt.close()

    ###########

    source_file_name = f'{base_directory}/{mode}/{year}/{month}/lw_source-{mode}-{year}-{month}.2.nc'
    dt = Dataset(source_file_name, "w")

    dim1 = dt.createDimension("col",col)
    dim2 = dt.createDimension("layers_and_surface",temp_layer.shape[1])
    dim3 = dt.createDimension("band",totplanck.shape[0])

    var = dt.createVariable("raw_sources","f4",("col","layers_and_surface","band"))
    var[:]=raw_sources[:]
    dt.close()

def examine_planck (planck_file_name="../../../../rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"):

    dt = Dataset(planck_file_name,'r')

    totplanck = dt.variables["totplnk"][:,:].data
    band = 16
    temperature_max = 50

    print(f"{totplanck[band-1,:temperature_max] / 340.0}")

    dt.close()

def examine_planck_2 (planck_file_name="/home/hws/rrtmg_lw.nc"):

    dt = Dataset(planck_file_name,'r')

    totplanck = dt.variables["IntegratedPlanckFunction"][:,:].data
    band = 10
    temperature_max = 50

    bandlowerlimit = dt.variables["BandWavenumberLowerLimit"][:].data
    bandupperlimit = dt.variables["BandWavenumberUpperLimit"][:].data

    TemperaturePlanckValues = dt.variables["TemperaturePlanckValues"][:].data

    print(f"Temp Planck = {TemperaturePlanckValues}")

    bandwidth = bandupperlimit - bandlowerlimit
    print(f"Bandwidth: {bandwidth}")

    #print(f"{totplanck[band,:temperature_max] / bandwidth[band]}")

    print(f"{totplanck[band,:temperature_max] }")

    dt.close()

def wrangle_ecrad_input_data(mode,month,year, base_directory):
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



def wrangle_nn_input_data(mode,month,year, base_directory):
    g = 9.80665
    m_co2 = 44.0095
    m_dry = 28.964
    m_o2 = 31.999
    m_n2o = 44.01280
    m_ch4 = 16.0425
    m_co = 28.010

    d = base_directory + f'{mode}/{year}/'  
    file_name_ecrad_input = d + f'{month}/lw_input-{mode}-{year}-{month}.nc'
    file_name_source_input = d + f'{month}/lw_source-{mode}-{year}-{month}.2.nc'
    file_name_flux_input = d + f'Flux_lw-{mode}-{year}-{month}.nc'
    file_name_nn_input = d + f'nn_input-{mode}-{year}-{month}.nc'
    dt_ecrad = Dataset(file_name_ecrad_input,"r")
    dt_source = Dataset(file_name_source_input,"r")
    dt_flux = Dataset(file_name_flux_input,"r")
    dt_nn = Dataset(file_name_nn_input,"w")
    temp_level = dt_ecrad.variables["temp_layer"][:,:,:].data
    pres_level = dt_ecrad.variables["pres_layer"][:,:,:].data

    shape = temp_level.shape
    col = shape[0] * shape[1]
    level = shape[2]
    dim_col = dt_nn.createDimension("col",col)
    dim_level = dt_nn.createDimension("level",level)
    dim_half_level = dt_nn.createDimension("half_level",level + 1)
    dim_two = dt_nn.createDimension("two",2)
    temp_level = temp_level.reshape((col,-1, 1))

    var_temp_pres_level = dt_nn.createVariable("temp_pres_level","f4",("col","level","two"))
    pres_level = pres_level.reshape((col,-1, 1))
    var_temp_pres_level[:,:,0] = temp_level[:,:,:]
    var_temp_pres_level[:,:,1] = pres_level[:,:,:]

    pres_half_level = dt_ecrad.variables["pressure_hl"][:,:].data
    delta_pressure = pres_half_level[:,1:] - pres_half_level[:,:-1]

    total_mass = (delta_pressure / g) 

    cw_mr = dt_ecrad.variables["q_liquid"][:,:].data # mixing ratio
    clwc = cw_mr / (1.0 + cw_mr)   # specific cloud water
    cw = clwc * total_mass

    ci_mr = dt_ecrad.variables["q_ice"][:,:].data
    ciwc = ci_mr / (1.0 + ci_mr)
    ci = ciwc * total_mass

    q = dt_ecrad.variables["q"][:,:].data
    water_vapor = q * total_mass
    water_vapor_mmr = q / (1.0 - q)
    if np.min(water_vapor_mmr) < 1.0e-09:
        print(f"small q {np.min(q)}")
        print(f"small mmr {np.min(water_vapor_mmr)}")
        dry_mass = total_mass
    else:
        dry_mass = water_vapor / water_vapor_mmr
        if np.isnan(np.sum(dry_mass)):
            print(f"dry_mass contains Nan")
            print(f"Indices of Nan = {np.argwhere(np.isnan(dry_mass))}")
            print(f"small mmr {np.min(water_vapor_mmr)}")
            os.abort()


    o3_mmr = dt_ecrad.variables["o3_mmr"][:,:].data
    o3 = o3_mmr * dry_mass
    if np.isnan(np.sum(o3_mmr)):
        print(f"o3_mmr contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(o3_mmr))}")
        os.abort()
    if np.isnan(np.sum(o3)):
        print(f"o3 contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(o3))}")
        os.abort()

    co2_vmr = dt_ecrad.variables["co2_vmr"][:,:].data
    co2 = dry_mass * co2_vmr * m_co2  / m_dry 
    if np.isnan(np.sum(co2_vmr)):
        print(f"co2_vmr contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(co2_vmr))}")
        os.abort()
    if np.isnan(np.sum(co2)):
        print(f"co2 contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(co2))}")
        os.abort()

    o2_vmr = dt_ecrad.variables["o2_vmr"][:].data
    o2 = dry_mass * o2_vmr * m_o2  / m_dry 
    if np.isnan(np.sum(o2_vmr)):
        print(f"o2_vmr contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(o2_vmr))}")
        os.abort()
    if np.isnan(np.sum(o2)):
        print(f"o2 contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(o2))}")
        os.abort()

    n2o_vmr = dt_ecrad.variables["n2o_vmr"][:,:].data
    n2o = dry_mass * n2o_vmr * m_n2o  / m_dry 
    if np.isnan(np.sum(n2o_vmr)):
        print(f"n2o_vmr contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(n2o_vmr))}")
        os.abort()
    if np.isnan(np.sum(n2o)):
        print(f"n2o contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(n2o))}")
        os.abort()

    ch4_vmr = dt_ecrad.variables["ch4_vmr"][:,:].data
    ch4 = dry_mass * ch4_vmr * m_ch4  / m_dry 
    if np.isnan(np.sum(ch4_vmr)):
        print(f"ch4_vmr contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(ch4_vmr))}")
        os.abort()
    if np.isnan(np.sum(ch4)):
        print(f"ch4 contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(ch4))}")
        os.abort()

    co_vmr = dt_ecrad.variables["co_vmr"][:,:].data
    co = dry_mass * co_vmr * m_co  / m_dry 
    if np.isnan(np.sum(co_vmr)):
        print(f"co_vmr contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(co_vmr))}")
        os.abort()
    if np.isnan(np.sum(co)):
        print(f"co contains Nan")
        print(f"Indices of Nan = {np.argwhere(np.isnan(co))}")
        os.abort()

    cw = cw.reshape((col,level, 1))
    ci = ci.reshape((col,level, 1))
    water_vapor = water_vapor.reshape((col,level, 1))
    o3 = o3.reshape((col,level, 1))
    co2 = co2.reshape((col,level, 1))

    o2 = o2.reshape((col,level, 1))
    n2o = n2o.reshape((col,level, 1))
    ch4 = ch4.reshape((col,level, 1))
    co = co.reshape((col,level, 1))

    constituents = np.concatenate((cw,ci,water_vapor,o3,co2,o2,n2o,ch4,co), axis=2)
    dim_feature = dt_nn.createDimension("feature",9)
    var_constituents = dt_nn.createVariable("constituents","f4",("col","level","feature"))
    var_constituents[:] = constituents[:]
    var_constituents.setncattr("description-1","mass")
    var_constituents.setncattr("description-2","liquid_water, ice_water, water_vapor, o3, co2, o2, n2o, ch4, co")

    emissivity = dt_ecrad.variables["lw_emissivity"][:].data
    lw_emissivity = dt_nn.createVariable("lw_emissivity","f4",("col",))
    lw_emissivity[:] = emissivity[:]

    var_delta = dt_nn.createVariable("delta_pressure","f4",("col","level",))
    var_delta[:] = delta_pressure[:]

    sources = dt_source.variables["raw_sources"][:,:,:].data
    dim_level_surface = dt_nn.createDimension("level_and_surface", sources.shape[1])
    dim_band = dt_nn.createDimension("band", sources.shape[2])
    var_sources = dt_nn.createVariable("sources","f4",("col","level_and_surface","band"))
    var_sources[:] = sources[:]

    flux_dn_lw = dt_flux.variables["flux_dn_lw"][:,:].data
    var_flux_dn_lw = dt_nn.createVariable("flux_dn_lw","f4",("col","half_level"))
    var_flux_dn_lw[:] = flux_dn_lw[:]

    flux_up_lw = dt_flux.variables["flux_up_lw"][:,:].data
    var_flux_up_lw = dt_nn.createVariable("flux_up_lw","f4",("col","half_level"))
    var_flux_up_lw[:] = flux_up_lw[:]

    flux_dn_lw_clear = dt_flux.variables["flux_dn_lw_clear"][:,:].data
    var_flux_dn_lw_clear = dt_nn.createVariable("flux_dn_lw_clear","f4",("col","half_level"))
    var_flux_dn_lw_clear[:] = flux_dn_lw_clear[:]

    flux_up_lw_clear = dt_flux.variables["flux_up_lw_clear"][:,:].data
    var_flux_up_lw_clear = dt_nn.createVariable("flux_up_lw_clear","f4",("col","half_level"))
    var_flux_up_lw_clear[:] = flux_up_lw_clear[:]

    dt_nn.close()
    dt_ecrad.close()
    dt_source.close()
    dt_flux.close()

def examine_nn_input_data(mode,month,year, base_directory):
    d = base_directory + f'{mode}/{year}/'  
    file_name_nn_input = d + f'nn_input-{mode}-{year}-{month}.nc'

    dt = Dataset(file_name_nn_input, "r")
    composition = dt.variables["constituents"][:,:,:].data
    min = np.min(composition, axis=(0,1))
    mean = np.mean(composition, axis=(0,1))
    max = np.max(composition, axis=(0,1))
    dt.close()

    print(f"min = {min}")
    print(f"mean = {mean}")
    print(f"max = {max}")

if __name__ == "__main__":


    base_directory = f'/data-T1/hws/CAMS/processed_data/'

    if False:
        mode = 'training'
        month = '02'
        year = '2008'

        #wrapper_raw_sources (mode,month,year, base_directory)
        #wrangle_nn_input_data(mode,month,year, base_directory)

        examine_nn_input_data(mode,month,year, base_directory)

    if True:
        months = [str(m).zfill(2) for m in range(1,13)]

        combo = [('training','2008'),('cross_validation','2008'),('testing','2009'),('testing','2015'),('testing','2020')]

        for c in combo:
            mode = c[0]
            year = c[1]
            print(f'Processing {mode} {year}')
            for month in months[:]:
                print(f'{year} {month}')
                #wrangle_ecrad_input_data(mode, month, year, base_directory)
                #wrapper_raw_sources (mode,month,year, base_directory)
                wrangle_nn_input_data(mode, month, year, base_directory)


    if False:
        examine_planck_2()


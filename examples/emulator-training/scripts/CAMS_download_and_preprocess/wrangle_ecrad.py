# run with: conda activate netcdf3.8
from netCDF4 import Dataset,num2date
import numpy as np
import xarray as xr
import os
import sys

def generate_raw_sources(totplanck, min_temp, max_temp, temp_layers, temp_skin, bandwidth):
    #double totplnk(bnd, temperature_Planck)
    # examples, layers, bands
    raw_sources = np.zeros((temp_layers.shape[0],temp_layers.shape[1], totplanck.shape[0]), dtype=np.float32)

    surface_source = np.zeros((temp_skin.shape[0], totplanck.shape[0]), dtype=np.float32)

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

    if True:
        print(f"Min allowable temp = {min_temp}. Actual min temp = {np.min(temp_skin)}")
        print(f"Max allowable temp = {max_temp}. Actual max temp = {np.max(temp_skin)}")

    if np.min(temp_skin) < min_temp:
        print(f"Out of range: Min allowable temp = {min_temp}. Actual min temp = {np.min(temp_skin)}")
        exit()

    if np.max(temp_skin) > max_temp:
        print(f"Out of range: Max allowable temp = {max_temp}. Actual max temp = {np.max(temp_skin)}")
        exit()

    for i in np.arange(temp_skin.shape[0]):
            diff = temp_skin[i] - min_temp
            index = np.int32(np.floor(diff))
            fraction = diff - np.float32(index)
            surface_source[i,:] = 1.0e04 * bandwidth[:] * np.pi * (totplanck[:,index] * (1.0 - fraction) + totplanck[:,index + 1] * fraction)

    return raw_sources, surface_source



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
    #temp_layer = dt.variables["temp_layer"][:,:,:].data
    temp_level = dt.variables["temperature_hl"][:,:].data
    shape = temp_level.shape

    col = shape[0]

    temp_skin = dt.variables["skin_temperature"][:].data
    temp_skin = temp_skin.reshape((-1,))

    level_sources, surface_source = generate_raw_sources(totplanck, min_temp, max_temp, temp_level, temp_skin, bandwidth)

    dt.close()

    ###########

    source_file_name = f'{base_directory}/{mode}/{year}/{month}/lw_source-{mode}-{year}-{month}.nc'
    dt = Dataset(source_file_name, "w")

    dim1 = dt.createDimension("col",col)
    dim2 = dt.createDimension("half_level",temp_level.shape[1])
    dim3 = dt.createDimension("band",totplanck.shape[0])

    var = dt.createVariable("half_level_sources","f4",("col","half_level","band"))
    var[:]=level_sources[:]

    var2 =  dt.createVariable("surface_source","f4",("col","band"))
    var2[:]=surface_source[:]

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
    data = np.reshape(data, (n_col,))
    data = np.cos(data * np.pi / 180.0)

    var2 = dt.createVariable("cos_solar_zenith_angle","f4",("col",))
    var2[:]= data[:]
    var2.setncattr("units","1")
    var2.setncattr("long_name", "Cosine of the solar zenith angle")

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


    # ecRad uses it as a mass ratio. See rrtm_prepare_gases.F90
    # see line 224 in particular.
    # Note however, that on line 188 it interprets it as
    # specific humidity to compute the molecular
    # weight of moist (total) air, but uses it as a mass ratio
    # in line 189 for computing the mass of dry-air

    # Do *NOT* convert from mass ratio to specific humidity
    # REMOVED: data = data / (1.0 + data)

    var5 = dt.createVariable("q","f4",("col","level"))
    var5[:] = data[:]
    var5.setncattr("units","1")
    var5.setncattr("long_name","Water vapor mass ratio")

    var5.setncattr("note_1","In practice ecRad treats this as a mass ratio")
    var5.setncattr("note_2","even though q usually indicates specific humidity")

    m_o3 = 47.99820

    data = dt.variables['ozone'][:,:,:].data
    data =np.reshape(data, (n_col,n_level))

    # converting from vmr to mass ratio
    data = data * m_o3 / m_dry

    var6 = dt.createVariable("o3_mmr","f4",("col","level"))
    var6[:] = data[:]
    var6.setncattr("units","1")
    var6.setncattr("long_name", "Ozone mass mixing ratio")

    #####
    new_data = np.full((n_col, n_level), 0.1, dtype=np.float32)
    var7 = dt.createVariable("cloud_fraction","f4",("col","level"))
    var7[:] = new_data[:]
    var7.setncattr("units", "1")
    var7.setncattr("long_name", "Cloud fraction")

    data = dt.variables['clwc'][:,:,:].data
    data = np.reshape(data, (n_col,n_level))

    # Do *NOT* convert from specific content to mass ratio
    #data = data / (1.0 - data)

    var8 = dt.createVariable("q_liquid","f4",("col","level"))
    var8[:]=data[:]
    var8.setncattr("units","1")
    var8.setncattr("name","clwc")
    var8.setncattr("long_name","Specific cloud liquid water content")
    var8.setncattr("Note-1","ecRad calls this the gridbox-mean liquid mixing ratio")
    var.setncattr("Note-2","but uses it as specific cloud liquid content")

    data = dt.variables['ciwc'][:,:,:].data
    data = np.reshape(data, (n_col,n_level))

    # Do *NOT* convert from specific content to mass ratio
    #data = data / (1.0 - data)

    var9 = dt.createVariable("q_ice","f4",("col","level"))
    var9[:]=data[:]
    var9.setncattr("units","1")
    var8.setncattr("name","ciwc")
    var9.setncattr("long_name","Specific cloud ice water content")
    var9.setncattr("Note-1","ecRad calls this the gridbox-mean ice mixing ratio")
    var9.setncattr("Note-2","but uses it as specific cloud ice content")

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

# changing rel and rei to match RTE-RRTMGP
# trying to reconcile ecrad scheme with RTE-RRTMGP
def transform_ecrad_input_data(mode,month,year, base_directory, is_just_o2=True, is_mcica=True):

    d = base_directory + f'{mode}/{year}/'  

    if is_mcica:
        file_name_ecrad = d + f'{month}/lw_input_mcica-{mode}-{year}-{month}'
    else:
        file_name_ecrad = d + f'{month}/lw_input-{mode}-{year}-{month}'

    cmd = f'cp {file_name_ecrad}.nc {file_name_ecrad}.tmp.nc'
    os.system(cmd)

    dt_ecrad = Dataset(file_name_ecrad + '.tmp.nc',"a")

    if is_just_o2:
        var_o2 = dt_ecrad.variables["oxygen_GM"]
        var_n2 = dt_ecrad.variables["nitrogen_GM"]

        o2 = np.full((var_o2.shape[0]),0.209)
        n2 = np.full((var_o2.shape[0]),0.7808)

        var_o2[:]= o2[:]
        var_n2[:]= n2[:]
    else:
        var_re_ice = dt_ecrad.variables["re_ice"]
        var_re_liquid = dt_ecrad.variables["re_liquid"]
        var_cloud_fraction = dt_ecrad.variables["cloud_fraction"]
        var_o2 = dt_ecrad.variables["oxygen_GM"]


        re_ice = np.full((var_re_ice.shape[0],var_re_ice.shape[1]),95.0e-06)
        re_liquid = np.full((var_re_ice.shape[0],var_re_ice.shape[1]),12.0e-06)
        cloud_fraction = np.full((var_cloud_fraction.shape[0],var_cloud_fraction.shape[1]),0.9999)
        o2 = np.full((var_o2.shape[0]),0.209)

        var_re_ice[:] = re_ice[:]
        var_re_liquid[:] = re_liquid[:]
        var_cloud_fraction[:] = cloud_fraction[:]
        var_o2[:]= o2[:]

    dt_ecrad.close()

# Adds fields for O2 and N2 to *.tmp.nc
def transform_rte_rrtmgp_input_data(mode,month,year, base_directory):

    d = base_directory + f'{mode}/{year}/'  
    file_name_rte_rrtmgp = d + f'{month}/CAMS_{year}-{month}.final.2'

    cmd = f'cp {file_name_rte_rrtmgp}.nc {file_name_rte_rrtmgp}.tmp.nc'
    os.system(cmd)

    dt = Dataset(file_name_rte_rrtmgp + '.tmp.nc',"a")

    var_o2 = dt.variables["oxygen_GM"]
    var_n2 = dt.variables["nitrogen_GM"]

    o2 = np.full((var_o2.shape[0]),0.209)
    n2 = np.full((var_o2.shape[0]),0.7808)

    var_o2[:]= o2[:]
    var_n2[:]= n2[:]

    dt.close()
    
def wrangle_lw_nn_input_data(mode,month,year, base_directory):
    # Using values from rrtm_prepare_gases.F90
    g = 9.80665 #
    m_co2 = 44.011 #
    m_dry = 28.970  # ZAMD
    m_h2o = 18.0154 # ZAMW
    m_o2 = 31.999
    m_n2o = 44.013 #
    m_ch4 = 16.043 #
    m_co = 28.010

    d = base_directory + f'{mode}/{year}/'  
    file_name_ecrad_input = d + f'{month}/lw_input-{mode}-{year}-{month}.nc'
    file_name_source_input = d + f'{month}/lw_source-{mode}-{year}-{month}.nc'
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

    clwc = dt_ecrad.variables["q_liquid"][:,:].data # specific liquid cloud water
    cw = clwc * total_mass

    ciwc = dt_ecrad.variables["q_ice"][:,:].data # specific ice cloud water
    ci = ciwc * total_mass

    # even though 'q' is variable name, this is the mass ratio
    water_vapor_mmr = dt_ecrad.variables["q"][:,:].data

    # Normally, dry mass would just be the following
    dry_mass = total_mass / (1.0 + water_vapor_mmr)
    # Does not for factor from line 188 in rrtmp_prepare_gases.F90
    q = water_vapor_mmr / (1.0 + water_vapor_mmr)
    water_vapor = q * total_mass

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

    half_level_sources = dt_source.variables["half_level_sources"][:,:,:].data
    #dim_level_surface = dt_nn.createDimension("level_and_surface", sources.shape[1])
    dim_band = dt_nn.createDimension("band", half_level_sources.shape[2])
    var_sources = dt_nn.createVariable("half_level_sources","f4",("col","half_level","band"))
    var_sources[:] = half_level_sources[:]

    surface_sources = dt_source.variables["surface_source"][:,:].data
    var_surface_sources = dt_nn.createVariable("surface_source","f4",("col","band"))
    var_surface_sources[:] = surface_sources[:]

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

def compute_delta_height_hypsometric(temp_level, pres_level, water_vapor_mmr):
    # Using Grant W. Petty - A First Course in Atmospheric Thermodynamics
    # Section 4.1.3. The hypsometric equation
    m_dry = 28.970  # ZAMD
    m_h2o = 18.0154 # ZAMW
    epsilon = m_h2o / m_dry
    g = 9.80665
    Rd = 287.058 # J kg^-1 K^-1
    
    virtual_temperature = temp_level * (
        (1.0 + water_vapor_mmr / epsilon) /
        (1.0 + water_vapor_mmr)
    )
    
    denominator = np.log(pres_level[:,1:]) - np.log(pres_level[:,:-1])
    numerator = temp_level[:,1:] * np.log(pres_level[:,1:]) - temp_level[:,:-1] * np.log(pres_level[:,:-1])
    mean_virtual_temperature = numerator / denominator
    delta_height = np.log(pres_level[:,1:]/pres_level[:,:-1]) * \
        mean_virtual_temperature  * Rd / g
        
    return delta_height
    
    
def wrangle_sw_nn_input_data(mode,month,year, base_directory, is_mcica=False, is_tripleclouds=False):
    # Using values from rrtm_prepare_gases.F90
    g = 9.80665 #
    m_co2 = 44.011 #
    m_dry = 28.970  # ZAMD
    m_h2o = 18.0154 # ZAMW
    m_o2 = 31.999
    m_n2o = 44.013 #
    m_ch4 = 16.043 #
    m_co = 28.010

    d = base_directory + f'{mode}/{year}/'  
    if mode == "validation":
        file_name_ecrad_input = d + f'{month}/lw_input_mcica-cross_{mode}-{year}-{month}.nc'
    else:
        file_name_ecrad_input = d + f'{month}/lw_input_mcica-{mode}-{year}-{month}.nc'
    file_name_old_input = d + f'Flux_sw-{year}-{month}.2.nc'

    if is_mcica:
        if mode == "validation":
            file_name_flux_input = d + f'Flux_lw_mcica-cross_{mode}-{year}-{month}.nc'
        else:
            file_name_flux_input = d + f'Flux_lw_mcica-{mode}-{year}-{month}.nc'
        file_name_nn_input = d + f'nn_input_sw_mcica-{mode}-{year}-{month}.nc'
        
    elif is_tripleclouds:

        file_name_flux_input = d + f'Flux_lw_tripleclouds-{mode}-{year}-{month}.2.nc'
        file_name_nn_input = d + f'nn_input_sw_tripleclouds-{mode}-{year}-{month}.nc'
    else:
        file_name_flux_input = d + f'Flux_lw-{mode}-{year}-{month}.nc'
        file_name_nn_input = d + f'nn_input_sw-{mode}-{year}-{month}.nc'

    dt_ecrad = Dataset(file_name_ecrad_input,"r")
    dt_flux = Dataset(file_name_flux_input,"r")
    dt_old = Dataset(file_name_old_input,"r")
    dt_nn = Dataset(file_name_nn_input,"w")

    temp_level = dt_ecrad.variables["temp_layer"][:,:,:].data
    pres_level = dt_ecrad.variables["pres_layer"][:,:,:].data
    
    cloud_fraction = dt_ecrad.variables["cloud_fraction"][:,:].data

    shape = temp_level.shape
    col = shape[0] * shape[1]
    level = shape[2]
    dim_col = dt_nn.createDimension("col",col)
    dim_level = dt_nn.createDimension("level",level)
    dim_half_level = dt_nn.createDimension("half_level",level + 1)
    dim_two = dt_nn.createDimension("two",2)

    mu = dt_old.variables["mu0"][:,:].data
    var_mu = dt_nn.createVariable("mu0","f4",("col",))
    var_mu.setncattr("long_name","Cosine of solar zenith angle")
    mu = mu.reshape((col,))
    var_mu[:]= mu[:]

    surface_albedo = dt_old.variables["sfc_alb"][:,:,:].data
    surface_albedo = surface_albedo[:,:,0]
    var_surface_albedo = dt_nn.createVariable("surface_albedo","f4",("col",))
    var_surface_albedo.setncattr("long_name","surface albedo")
    surface_albedo = surface_albedo.reshape((col,))
    var_surface_albedo[:]= surface_albedo[:]

    is_valid = dt_old.variables["is_valid_zenith_angle"][:,:].data
    var_is_valid = dt_nn.createVariable("is_valid_zenith_angle","f4",("col",))
    var_is_valid.setncattr("long_name","True if zenith angle is less than 90 degrees")
    is_valid = is_valid.reshape((col,))
    var_is_valid[:]= is_valid[:]

    temp_level = temp_level.reshape((col,-1))

    var_temp_pres_level = dt_nn.createVariable("temp_pres_level","f4",("col","level","two"))
    pres_level = pres_level.reshape((col,-1))
    var_temp_pres_level[:,:,0] = temp_level[:,:]
    var_temp_pres_level[:,:,1] = pres_level[:,:]

    pres_half_level = dt_ecrad.variables["pressure_hl"][:,:].data
    delta_pressure = pres_half_level[:,1:] - pres_half_level[:,:-1]

    #var_pres_half_level = dt_nn.createVariable("pressure_half_level","f4",("col","half_level",))
    #var_pres_half_level[:] = pres_half_level

    total_mass = (delta_pressure / g) 

    clwc = dt_ecrad.variables["q_liquid"][:,:].data # specific liquid cloud water
    cw = clwc * total_mass * 1000.0 # converting to g / kg; consistent with original version

    ciwc = dt_ecrad.variables["q_ice"][:,:].data # specific ice cloud water
    ci = ciwc * total_mass  * 1000.0 # converting to g / kg; consistent with original version

    # even though 'q' is variable name, this is the mass ratio
    water_vapor_mmr = dt_ecrad.variables["q"][:,:].data
    
    # Normally, dry mass would just be the following
    dry_mass = total_mass / (1.0 + water_vapor_mmr)
    # Does not for factor from line 188 in rrtmp_prepare_gases.F90
    q = water_vapor_mmr / (1.0 + water_vapor_mmr)
    water_vapor = q * total_mass
    
    if is_mcica or is_tripleclouds:
        var_cf = dt_nn.createVariable("cloud_fraction","f4",("col","level"))
        var_cf[:] = cloud_fraction[:]
        var_cf.setncattr("description","cloud fraction: 0.0 <= cloud_fraction <= 1.0")
        
        var_wv = dt_nn.createVariable("wv","f4",("col","level"))
        var_wv[:] = water_vapor_mmr[:]
        var_wv.setncattr("description","water vapor mass-mixing ratio (mmr) per dry air")
        
        var_wl = dt_nn.createVariable("wl","f4",("col","level"))
        var_wl[:] = clwc[:] * total_mass / dry_mass
        var_wl.setncattr("description","cloud liquid water mass-mixing ratio (mmr) per dry air")
        
        var_wi = dt_nn.createVariable("wi","f4",("col","level"))
        var_wi[:] = ciwc[:] * total_mass[:] / dry_mass[:]
        var_wi.setncattr("description","cloud ice water mass-mixing ratio (mmr) per dry air")
        
        # Bolton (1980) Formula
        t_celsius = temp_level - 273.15
        e_sat = 611.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))  # Pa
        epsilon = m_h2o / m_dry
        var_wsat = dt_nn.createVariable("wsat","f4",("col","level"))
        var_wsat[:] = epsilon * e_sat / (pres_level - e_sat)
        var_wsat.setncattr("description","saturaton mass-mixing ratio (mmr) per dry air")
        
        var_dry_mass = dt_nn.createVariable("dry_mass","f4",("col","level"))
        var_dry_mass[:] = dry_mass[:]
        var_dry_mass.setncattr("description","Dry Mass (kg)")
        
        dim_level_differences = dt_nn.createDimension("level_differences",level - 1)
        var_delta_height = dt_nn.createVariable("delta_height","f4",("col","level_differences"))
        var_delta_height[:] = compute_delta_height_hypsometric(temp_level, pres_level, water_vapor_mmr)
        var_delta_height.setncattr("description","Geometric distance between layers")

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


    #o2_vmr = dt_ecrad.variables["o2_vmr"][:].data
    o2_vmr = np.array([0.2095])
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
    #co = co.reshape((col,level, 1))

    constituents = np.concatenate((cw,ci,water_vapor,o3,co2,o2,n2o,ch4), axis=2)
    dim_feature = dt_nn.createDimension("feature",8)
    var_constituents = dt_nn.createVariable("constituents","f4",("col","level","feature"))
    var_constituents[:] = constituents[:]
    var_constituents.setncattr("description-1","mass")
    var_constituents.setncattr("description-2","liquid_water, ice_water, water_vapor, o3, co2, o2, n2o, ch4")

    var_delta = dt_nn.createVariable("delta_pressure","f4",("col","level",))
    var_delta[:] = delta_pressure[:]

    flux_down = dt_flux.variables['flux_dn_sw'][:,:].data
    flux_up = dt_flux.variables['flux_up_sw'][:,:].data
    flux_down_direct = dt_flux.variables['flux_dn_direct_sw'][:,:].data
    flux_down_clear = dt_flux.variables['flux_dn_sw_clear'][:,:].data
    flux_down_direct_clear = dt_flux.variables['flux_dn_direct_sw_clear'][:,:].data
    flux_up_clear = dt_flux.variables['flux_up_sw_clear'][:,:].data

    var_flux_down_direct = dt_nn.createVariable("flux_down_direct","f4",("col","half_level"))
    var_flux_down_direct[:] = flux_down_direct[:]

    var_flux_down_diffuse = dt_nn.createVariable("flux_down_diffuse","f4",("col","half_level"))
    var_flux_down_diffuse[:] = flux_down[:] - flux_down_direct[:] 

    var_flux_up_diffuse = dt_nn.createVariable("flux_up_diffuse","f4",("col","half_level"))
    var_flux_up_diffuse[:] = flux_up[:]

    var_flux_down_direct_clear = dt_nn.createVariable("flux_down_direct_clear","f4",("col","half_level"))
    var_flux_down_direct_clear[:] = flux_down_direct_clear[:]

    var_flux_down_diffuse_clear = dt_nn.createVariable("flux_down_diffuse_clear","f4",("col","half_level"))
    var_flux_down_diffuse_clear[:] = flux_down_clear[:] - flux_down_direct_clear[:]

    var_flux_up_diffuse_clear = dt_nn.createVariable("flux_up_diffuse_clear","f4",("col","half_level"))
    var_flux_up_diffuse_clear[:] = flux_up_clear[:]

    dt_nn.close()
    dt_ecrad.close()
    dt_flux.close()
    dt_old.close()


def wrangle_sw_ukkonen_input_data(mode,month,year, base_directory):
    # Use values for gas, pressure, temp inputs from original file

    d = base_directory + f'{mode}/{year}/'  

    file_name_flux_input = d + f'Flux_lw-{mode}-{year}-{month}.nc'

    file_name_old_input = d + f'Flux_sw-{year}-{month}.2.nc'
    file_name_ukkonen_input = d + f'Flux_Ukkonen-{year}-{month}.nc'

    cmd = f'cp {file_name_old_input} {file_name_ukkonen_input}'
    os.system(cmd)

    o1 = d + "tmp1.nc"

    # Removes these variables
    cmd = f'ncks -x -v rsu,rsd,rsd_dir {file_name_ukkonen_input} {o1}'
    os.system(cmd)

    cmd = f'mv -f {o1} {file_name_ukkonen_input}'
    os.system(cmd)

    if True:
        dt_flux = Dataset(file_name_flux_input,"r")
        dt_nn = Dataset(file_name_ukkonen_input,"a")

        pres_level = dt_nn.variables["pres_level"][:,:,:].data
        shape = pres_level.shape

        rsu = dt_flux.variables['flux_up_sw'][:,:].data
        rsu = rsu.reshape((shape[0],shape[1],shape[2]))

        var_rsu = dt_nn.createVariable("rsu","f4",("expt","site","level"))
        var_rsu.setncattr("long_name","upwelling shortwave flux")
        var_rsu[:]= rsu[:]

        rsd = dt_flux.variables['flux_dn_sw'][:,:].data
        rsd = rsd.reshape((shape[0],shape[1],shape[2]))

        var_rsd = dt_nn.createVariable("rsd","f4",("expt","site","level"))
        var_rsd.setncattr("long_name","downwelling shortwave flux")
        var_rsd[:]= rsd[:]

        rsd_dir = dt_flux.variables['flux_dn_direct_sw'][:,:].data
        rsd_dir = rsd_dir.reshape((shape[0],shape[1],shape[2]))

        var_rsd_dir = dt_nn.createVariable("rsd_dir","f4",("expt","site","level"))
        var_rsd_dir.setncattr("long_name","downwelling direct shortwave flux")
        var_rsd_dir[:]= rsd_dir[:]

        dt_nn.close()
        dt_flux.close()

def wrangle_openbox_to_ukkonen_input_data(mode,month,year, base_directory):
    g = 9.80665 #
    m_co2 = 44.011 #
    m_dry = 28.970  # ZAMD
    m_h2o = 18.0154 # ZAMW
    m_o2 = 31.999
    m_o3 = 47.9985
    m_n2o = 44.013 #
    m_ch4 = 16.043 #
    m_co = 28.010
    n_sites = 5120
    
    d = base_directory + f'{mode}/{year}/'  
    file_name_openbox = d + f'shortwave-{mode}-{year}-{month}.nc'
    file_name_ukkonen = d + f'shortwave-{mode}-ukkonen-format-{year}-{month}.nc'

    dt_openbox = Dataset(file_name_openbox,"r")
    dt_ukkonen = Dataset(file_name_ukkonen,"w")
    
    constituents = dt_openbox.variables['constituents'][:,:,:].data
    n_features = constituents.shape[2]
    
    rsu = dt_openbox.variables['flux_up_diffuse'][:,:].data
    shape = rsu.shape
    expt = np.int32(shape[0]/n_sites)
    
    dt_ukkonen.createDimension("site", n_sites)
    dt_ukkonen.createDimension("expt", expt)
    dt_ukkonen.createDimension("layer", shape[1]-1)
    dt_ukkonen.createDimension("level", shape[1])
    dt_ukkonen.createDimension("feature", n_features-1)

    delta_pressure = dt_openbox.variables["delta_pressure"][:,:].data
    delta_pressure = delta_pressure.reshape((expt, n_sites, shape[1]-1))
    var_delta_pressure = dt_ukkonen.createVariable("delta_pressure","f4",("expt","site","layer"))
    var_delta_pressure[:] = delta_pressure[:]
    
    total_mass = (delta_pressure / g) 
    total_mass = np.expand_dims(total_mass,axis=3)

    rsu = rsu.reshape((expt, n_sites, shape[1]))
    var_rsu = dt_ukkonen.createVariable("rsu","f4",("expt","site","level"))
    var_rsu.setncattr("long_name","upwelling shortwave flux")
    var_rsu[:]= rsu[:]

    rsd_direct = dt_openbox.variables['flux_down_direct'][:,:].data
    rsd_diffuse = dt_openbox.variables['flux_down_diffuse'][:,:].data
    rsd = rsd_direct + rsd_diffuse
    rsd = rsd.reshape((expt, n_sites, shape[1]))
    var_rsd = dt_ukkonen.createVariable("rsd","f4",("expt","site","level"))
    var_rsd.setncattr("long_name","downwelling shortwave flux")
    var_rsd[:]= rsd[:]

    rsd_direct = rsd_direct.reshape((expt, n_sites, shape[1]))
    var_rsd_dir = dt_ukkonen.createVariable("rsd_dir","f4",("expt","site","level"))
    var_rsd_dir.setncattr("long_name","downwelling direct shortwave flux")
    var_rsd_dir[:]= rsd_direct[:]
    
    # Masses of the constituents
    constituents = constituents.reshape((expt, n_sites, shape[1]-1,n_features))

    lwp = constituents[:,:,:,0]
    iwp = constituents[:,:,:,1]
    
    var_lwp = dt_ukkonen.createVariable("cloud_lwp","f4",("expt","site","layer"))
    var_lwp.setncattr("long_name","cloud liquid water path")
    var_lwp.setncattr("units","g/kg")
    var_lwp[:] = lwp[:]

    var_iwp = dt_ukkonen.createVariable("cloud_iwp","f4",("expt","site","layer"))
    var_iwp.setncattr("long_name","cloud liquid water path")
    var_iwp.setncattr("units","g/kg")
    var_iwp[:] = iwp[:]
    
    is_valid_zenith_angle = dt_openbox.variables['is_valid_zenith_angle'][:].data
    is_valid_zenith_angle = is_valid_zenith_angle.reshape((expt,n_sites))
    var_is_valid_zenith_angle = dt_ukkonen.createVariable("is_valid_zenith_angle","f4",("expt","site"))
    var_is_valid_zenith_angle.setncattr("long_name","True if zenith angle is less than 90 degrees")
    var_is_valid_zenith_angle[:] = is_valid_zenith_angle[:]
    
    mu0 = dt_openbox.variables['mu0'][:].data
    mu0 = mu0.reshape((expt,n_sites))
    var_mu0 = dt_ukkonen.createVariable("mu0","f4",("expt","site"))
    var_mu0.setncattr("long_name","cosine of solar zenith angle")
    var_mu0[:] = mu0[:]
    
    temp_pressure = dt_openbox.variables['temp_pres_level'][:,:,:].data
    temp_pressure = temp_pressure.reshape((expt,n_sites,shape[1]-1,2))
    
    # Compute mass ratios: mass_of_constituent / mass_of_dry_air
    r = constituents[:,:,:,2:] / (total_mass - constituents[:,:,:,2:])
        
    # Transform to volume ratios
    m_mass = [m_h2o/m_dry, m_o3/m_dry, m_co2/m_dry, m_o2/m_dry, m_n2o/m_dry, m_ch4/m_dry] 
    r = r / m_mass
    
    rrtmgp_sw_input = np.concatenate((temp_pressure,r[:,:,:,0:3], r[:,:,:,4:6]), axis=3)
    
    var_rrtmgp_sw_input = dt_ukkonen.createVariable("rrtmgp_sw_input","f4",("expt","site","layer","feature"))
    var_rrtmgp_sw_input.setncattr("long_name","inputs for RRTMGP shortwave gas optics")
    var_rrtmgp_sw_input.setncattr("comments","Features: tlay play h2o o3 co2 n2o ch4")
    var_rrtmgp_sw_input[:] = rrtmgp_sw_input[:]
    
    surface_albedo = dt_openbox.variables['surface_albedo'][:].data
    surface_albedo = surface_albedo.reshape((expt,n_sites))
    var_surface_albedo = dt_ukkonen.createVariable("sfc_alb","f4",("expt","site"))
    var_surface_albedo.setncattr("long_name","surface albedo")
    var_surface_albedo[:] = surface_albedo[:]
    
    dt_openbox.close()
    dt_ukkonen.close()
    
def compare_ukkonen_input_data(mode,month,year, base_directory):
    g = 9.80665 #
    m_co2 = 44.011 #
    m_dry = 28.970  # ZAMD
    m_h2o = 18.0154 # ZAMW
    m_o2 = 31.999
    m_n2o = 44.013 #
    m_ch4 = 16.043 #
    m_co = 28.010
    
    d = base_directory + f'{mode}/{year}/'  
    file_name_ukkonen_1 = d + f'shortwave-{mode}-{year}-{month}-ukkonen_format.nc'
    dt_1 = Dataset(file_name_ukkonen_1,"r")
    file_name_ukkonen_2 = d + f'shortwave-ukkonen-format-{mode}-{year}-{month}.nc'
    dt_2 = Dataset(file_name_ukkonen_2,"r")
    
    file_name_ecrad_input = d + f'{month}/lw_input-{mode}-{year}-{month}.nc'
    
    dt_3 = Dataset(file_name_ecrad_input,"r")
    
    c_1 = dt_1.variables['rrtmgp_sw_input'][:,:,:,:].data
    
    c_2 = dt_2.variables['rrtmgp_sw_input'][:,:,:,:].data

    diff = c_1[:,:,:,0] - c_2[:,:,:,0]
    
    print(f'temp min = {np.min(diff)}')
    print(f'temp max = {np.max(diff)}')
    
    diff = c_1[:,:,:,1] - c_2[:,:,:,1]
    
    print(f'pres min = {np.min(diff)}')
    print(f'pres max = {np.max(diff)}')
    
    diff = c_1[:,:,1:,2] - c_2[:,:,1:,2]
    sum = c_1[:,:,1:,2] + c_2[:,:,1:,2]
    
    print(f'h2o min = {np.min(diff)}')
    print(f'h2o max = {np.max(diff)}')
    

    print(f'h2o norm max = {np.max(diff / sum)}')
    
    eps = 0.000000000000001
    #diff = c_1[:,:,1:,2] - c_2[:,:,1:,2]
    diff = c_1[:,:,:,2] / (c_2[:,:,:,2] + eps)
    
    print(f'h2o min = {np.min(diff)}')
    print(f'h2o max = {np.max(diff)}')
    

    diff = c_1[:,:,1:,3] / (c_2[:,:,1:,3] + eps)
    
    print(f'o3 min = {np.min(diff)}')
    print(f'o3 max = {np.max(diff)}')
    
    eps1 = np.max(c_2[:,:,1:,4]) * 0.00000000001
    
    diff = c_1[:,:,:,4] - c_2[:,:,:,4]
    
    print(f'co2 min = {np.min(diff)}')
    print(f'co2 max = {np.max(diff)}')
    
    diff = c_1[:,:,1:,4] / (c_2[:,:,1:,4] + eps1)
    
    print(f'co2 ratio min = {np.min(diff)}')
    print(f'co2 ratio max = {np.max(diff)}')
    
    diff = c_1[:,:,:,5] - c_2[:,:,:,5]
    
    print(f'n2o min = {np.min(diff)}')
    print(f'n2o max = {np.max(diff)}')
    eps1 = np.max(c_2[:,:,1:,5]) * 0.00000000001
    diff = c_1[:,:,1:,5] / (c_2[:,:,1:,5] + eps1)
    
    print(f'n2o ratio min = {np.min(diff)}')
    print(f'n2o ratio max = {np.max(diff)}')
    
    diff = c_1[:,:,:,6] - c_2[:,:,:,6]
    
    print(f'ch4 min = {np.min(diff)}')
    print(f'ch4 max = {np.max(diff)}')
    eps1 = np.max(c_2[:,:,1:,6]) * 0.00000000001
    diff = c_1[:,:,1:,6] / (c_2[:,:,1:,6] + eps1)
    
    print(f'ch4 ratio min = {np.min(diff)}')
    print(f'ch4 ratio max = {np.max(diff)}')
    
    c_1 = dt_1.variables['cloud_lwp'][:,:,:].data
    
    c_2 = dt_2.variables['cloud_lwp'][:,:,:].data
    
    diff = c_1[:,:,:] - c_2[:,:,:]
    print(f'lwp min = {np.min(diff)}')
    print(f'lwp max = {np.max(diff)}')
    
    c_1 = dt_1.variables['cloud_iwp'][:,:,:].data
    
    c_2 = dt_2.variables['cloud_iwp'][:,:,:].data
    
    diff = c_1[:,:,:] - c_2[:,:,:]
    print(f'iwp min = {np.min(diff)}')
    print(f'iwp max = {np.max(diff)}')
    
    c_1 = dt_1.variables['pres_level'][:,:,:].data
    
    print(f'Min pres_level = {np.min(c_1[:,:,:])}')
    print(f'Max pres_level = {np.max(c_1[:,:,:])}')
    
    print(f'Min pres_level 0 = {np.min(c_1[:,:,0])}')
    print(f'Max pres_level 0  = {np.max(c_1[:,:,0])}')
    
    print(f'Min pres_level 1 = {np.min(c_1[:,:,1])}')
    print(f'Mean pres_level 1 = {np.mean(c_1[:,:,1])}')
    print(f'Max pres_level 1  = {np.max(c_1[:,:,1])}')
    
    c_1 = c_1[:,:,1:] - c_1[:,:,:-1]
    #c_1 = c_1[:,:,:-1] - c_1[:,:,1:]
    
    c_2 = dt_2.variables['delta_pressure'][:,:,:].data
    
    c_3 = dt_3.variables['pressure_hl'][:,:].data
    
    print(f'Min pressure_hl = {np.min(c_3[:,0])}')
    print(f'Max pressure_hl = {np.max(c_3[:,0])}')
    
    shape = c_1.shape
    
    c_3 = c_3.reshape((shape[0],shape[1],shape[2] + 1))
    
    c_3 = c_3[:,:,1:] - c_3[:,:,:-1]
    
    diff = c_1[:,:,:] - c_2[:,:,:]
    print(f'dp min = {np.min(diff)}')
    print(f'dp max = {np.max(diff)}')
    
    diff = c_1[:,:,1:] - c_2[:,:,1:]
    print(f'dp all levels except 0 min = {np.min(diff)}')
    print(f'dp all levels except 0 max = {np.max(diff)}')
    
    diff = c_1[:,:,-1] - c_2[:,:,-1]
    print(f'dp min = {np.min(diff)}')
    print(f'dp max = {np.max(diff)}')
    
    diff = c_1[:,:,:] - c_3[:,:,:]
    print(f'c3 dp min = {np.min(diff)}')
    print(f'c3 dp max = {np.max(diff)}')
    
    diff = c_1[:,:,0] - c_3[:,:,0]
    print(f'dp min = {np.min(diff)}')
    print(f'dp max = {np.max(diff)}')
    
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
    
def add_random_seed (mode,month,year, base_directory):
    d = base_directory + f'{mode}/{year}/'  
    file_name_input = d + f'{month}/lw_input_mcica-{mode}-{year}-{month}.nc'
    
    file_name_output = d + f'{month}/lw_input_mcica-{mode}-{year}-{month}.2.nc'
    
    ds_output = Dataset(file_name_output, "a")
    sw_albedo = ds_output.variables["sw_albedo"][:].data
    random_seed =np.ones((sw_albedo.shape[0],))
    
    var_random_seed = ds_output.createVariable("iseed","f4",("col",))
    var_random_seed[:] = random_seed[:]
    
    ds_output.close()
    
    
    
    
    
    

def compute_ecrad_output_data(mode,month,year, base_directory, is_mcica=False, is_tmp=False, is_tripleclouds=False):
    d = base_directory + f'{mode}/{year}/'  
    # .tmp.nc contains the updated liquid and ice radii
    if is_mcica:
        # Note that lw_input-{mode} and lw_input_mcica-{mode} only differ
        # in the cloud_fraction field. The mcica version has credible values
        # for this field
        if is_tmp:
            file_name_input = d + f'{month}/lw_input_mcica-{mode}-{year}-{month}.tmp.nc'
            file_name_output = d + f'Flux_lw_mcica-{mode}-{year}-{month}.tmp.nc'
        else:
            if mode == "validation":
                file_name_input = d + f'{month}/lw_input_mcica-cross_{mode}-{year}-{month}.nc'
            else:
                file_name_input = d + f'{month}/lw_input_mcica-{mode}-{year}-{month}.nc'
            #file_name_input = d + f'{month}/lw_input_mcica-{mode}-{year}-{month}.2.nc' # Uses a different random seed
            if is_tripleclouds:
                file_name_output = d + f'Flux_lw_tripleclouds-{mode}-{year}-{month}.2.nc'
                ex = '/home/hws/ecrad/bin/ecrad_working /home/hws/ecrad/practical/config.4.nam'
            else:
                file_name_output = d + f'Flux_lw_mcica-{mode}-{year}-{month}.2.nc'
                ex = '/home/hws/ecrad/bin/ecrad_working /home/hws/ecrad/practical/config.3.nam'

    else:
        if is_tmp:
            file_name_input = d + f'{month}/lw_input-{mode}-{year}-{month}.tmp.nc'
            file_name_output = d + f'Flux_lw-{mode}-{year}-{month}.tmp.nc'
        else:
            file_name_input = d + f'{month}/lw_input-{mode}-{year}-{month}.nc'
            file_name_output = d + f'Flux_lw-{mode}-{year}-{month}.nc'
        ex = '/home/hws/ecrad/bin/ecrad_hws /home/hws/ecrad/practical/config.2.nam'

    cmd = f'{ex} {file_name_input} {file_name_output}'
    print (cmd)
    os.system(cmd)

def compare_ecrad_with_rte_rrtmgp(mode,month,year, base_directory):
    d = base_directory + f'{mode}/{year}/'  
    file_name_rr = d + f'Flux_sw-{year}-{month}.3.tmp.nc'
    file_name_ecrad = d + f'Flux_lw-{mode}-{year}-{month}.working.3.nc'

    file_name_ecrad_input = d + f'{month}/lw_input-{mode}-{year}-{month}.tmp.nc'

    dt_ecrad = Dataset(file_name_ecrad, "r")
    dt_rr = Dataset(file_name_rr, "r")
    dt_input = Dataset(file_name_ecrad_input, "r")

    flux_down_ecrad = dt_ecrad.variables['flux_dn_sw'][:,:].data

    flux_up_ecrad = dt_ecrad.variables['flux_up_sw'][:,:].data

    pressure_ecrad = dt_ecrad.variables["pressure_hl"][:,:].data

    flux_down_rr = dt_rr.variables['rsd'][:,:,:].data
    flux_up_rr = dt_rr.variables['rsu'][:,:,:].data
    shape = flux_down_rr.shape

    pressure_rr = dt_rr.variables["pres_level"][:,:,:].data

    mu_rr = dt_rr.variables["mu0"][:,:].data

    selection = dt_rr.variables['is_valid_zenith_angle'][:,:].data.astype(int)
    selection = np.reshape(selection, (shape[0]*shape[1]))
    selection = selection.astype(bool)

    flux_down_rr = flux_down_rr.reshape((shape[0]*shape[1], shape[2]))
    flux_up_rr = flux_up_rr.reshape((shape[0]*shape[1], shape[2]))

    pressure_rr = pressure_rr.reshape((shape[0]*shape[1], shape[2]))

    mu_ecrad = dt_input.variables['cos_solar_zenith_angle'][:].data

    mu_ecrad = mu_ecrad[selection]

    mu_rr = mu_rr.reshape((shape[0]*shape[1]))
    mu_rr = mu_rr[selection]

    flux_ecrad = np.concatenate((flux_down_ecrad[selection,:], flux_up_ecrad[selection,:]),axis=1)

    #flux_ecrad = flux_ecrad * 1412.0 / 1361.0 

    flux_rr = np.concatenate((flux_down_rr[selection,:], flux_up_rr[selection,:]),axis=1)

    flux_loss = np.sqrt(np.mean(np.square(flux_ecrad - flux_rr), axis=(0,1)))

    flux_bias = np.mean(flux_ecrad - flux_rr, axis=(0,1))

    pressure_loss = np.sqrt(np.mean(np.square(pressure_ecrad - pressure_rr), axis=(0,1)))

    mu_loss = np.sqrt(np.mean(np.square(mu_ecrad - mu_rr), axis=(0,)))

    print(f'Year = {year}, month={month} rmse flux = {flux_loss} flux bias = {flux_bias} pressure_loss = {pressure_loss} mu_loss = {mu_loss}')

    dt_rr.close()
    dt_ecrad.close()
    dt_input.close()
def compare_rte_rrtmgp_with_new_radiation(mode,month,year, base_directory):
    d = base_directory + f'{mode}/{year}/'  
    file_name_rr = d + f'Flux_sw-{year}-{month}.nc'
    file_name_new = d + f'Flux_sw-{year}-{month}.3.tmp.nc'

    dt_new = Dataset(file_name_new, "r")
    dt_rr = Dataset(file_name_rr, "r")

    flux_down_new = dt_new.variables['rsd'][:,:,:].data

    flux_up_new = dt_new.variables['rsu'][:,:,:].data

    flux_down_rr = dt_rr.variables['rsd'][:,:,:].data
    flux_up_rr = dt_rr.variables['rsu'][:,:,:].data
    shape = flux_down_rr.shape

    selection = dt_rr.variables['is_valid_zenith_angle'][:,:].data.astype(int)
    selection = np.reshape(selection, (shape[0]*shape[1]))
    selection = selection.astype(bool)

    flux_down_rr = flux_down_rr.reshape((shape[0]*shape[1], shape[2]))
    flux_up_rr = flux_up_rr.reshape((shape[0]*shape[1], shape[2]))

    flux_down_new = flux_down_new.reshape((shape[0]*shape[1], shape[2]))
    flux_up_new = flux_up_new.reshape((shape[0]*shape[1], shape[2]))

    flux_new = np.concatenate((flux_down_new[selection,:], flux_up_new[selection,:]),axis=1)

    #flux_new = flux_new * 1412.0 / 1361.0 

    flux_rr = np.concatenate((flux_down_rr[selection,:], flux_up_rr[selection,:]),axis=1)

    flux_loss = np.sqrt(np.mean(np.square(flux_new - flux_rr), axis=(0,1)))

    flux_bias = np.mean(flux_new - flux_rr, axis=(0,1))

    print(f'Year = {year}, month={month} rmse flux = {flux_loss} flux bias = {flux_bias}')

    dt_rr.close()
    dt_new.close()

def compare_ecrad_with_new_radiation(mode,month,year, base_directory, is_tmp=False):
    d = base_directory + f'{mode}/{year}/'  
    if is_tmp:
        file_name_new = d + f'Flux_lw_tripleclouds-{mode}-{year}-{month}.nc'
        file_name_ecrad = d + f'Flux_lw_mcica-{mode}-{year}-{month}.nc'
    else:
        file_name_new = d + f'Flux_lw_mcica-{mode}-{year}-{month}.nc'
        file_name_ecrad = d + f'Flux_lw-{mode}-{year}-{month}.nc'

    dt_ecrad = Dataset(file_name_ecrad, "r")
    dt_new = Dataset(file_name_new, "r")

    # longwave
    flux_down_ecrad = dt_ecrad.variables['flux_dn_lw'][:,:].data
    flux_up_ecrad = dt_ecrad.variables['flux_up_lw'][:,:].data

    flux_down_new = dt_new.variables['flux_dn_lw'][:,:].data
    flux_up_new = dt_new.variables['flux_up_lw'][:,:].data

    flux_ecrad = np.concatenate((flux_down_ecrad, flux_up_ecrad),axis=1)
    flux_new = np.concatenate((flux_down_new, flux_up_new),axis=1)

    flux_loss = np.sqrt(np.mean(np.square(flux_ecrad - flux_new), axis=(0,1)))

    flux_bias = np.mean(flux_ecrad - flux_new, axis=(0,1))

    print(f'Longwave: Year = {year}, month={month} rmse flux = {flux_loss} flux bias = {flux_bias}')

    # shortwave
    flux_down_ecrad = dt_ecrad.variables['flux_dn_sw'][:,:].data
    flux_up_ecrad = dt_ecrad.variables['flux_up_sw'][:,:].data

    flux_down_new = dt_new.variables['flux_dn_sw'][:,:].data
    flux_up_new = dt_new.variables['flux_up_sw'][:,:].data

    flux_ecrad = np.concatenate((flux_down_ecrad, flux_up_ecrad),axis=1)
    flux_new = np.concatenate((flux_down_new, flux_up_new),axis=1)

    flux_loss = np.sqrt(np.mean(np.square(flux_ecrad - flux_new), axis=(0,1)))

    flux_bias = np.mean(flux_ecrad - flux_new, axis=(0,1))

    print(f'Shortwave Year = {year}, month={month} rmse flux = {flux_loss} flux bias = {flux_bias}')

    dt_new.close()
    dt_ecrad.close()

def saturation_vapor_pressure(t):
    # Clausius-Clapeyron equation for water only
    # https://geo.libretexts.org/Bookshelves/Meteorology_and_Climate_Science/Practical_Meteorology_(Stull)/04%3A_Water_Vapor/4.00%3A_Vapor_Pressure_at_Saturation
    rd = 2.8705e2 # gas constant of dry air [J/kg/K]
    rv = 4.6150e2 # gas constant water [J/kg/K]
    lv = 2.5e6 # Latent haet of vaporization [J / Kg]
    ld = 2.83e6 # Latent heat of deposition [J / Kg]
    e0 = 611.3 #[Pa]
    t0 = 273.15 # [K]
    return e0 * np.exp((lv / rv) * ((1.0/t0) - (1.0/t)))
    #return e0 * np.exp((ld / rv) * ((1.0/t0) - (1.0/t)))    

def saturation_vapor_pressure_2(T):
    """
    Created on Mon Sep 13 15:01:22 2021

    @author: peter
    """
    ''' get sateration pressure (units [Pa]) for a given air temperature (units [K])'''
    TK = 273.15
    e1 = 101325.0
    logTTK = np.log10(T/TK)
    esat =  e1*10**(10.79586*(1-TK/T)-5.02808*logTTK+ 1.50474*1e-4*(1.-10**(-8.29692*(T/TK-1)))+ 0.42873*1e-3*(10**(4.76955*(1-TK/T))-1)-2.2195983) 
    return esat


def relative_humidity(w, es, p):

    # https://vortex.plymouth.edu/~stmiller/stmiller_content/Publications/AtmosRH_Equations_Rev.pdf eq 3
    m_dry = 28.9644 # more signficant digits
    m_h2o =  18.01528
    e = w * p / ((m_h2o / m_dry) + w) #correct
    #e = w * rv * p / (rd + w * rv)
    rh = e / es
    return rh

def saturation_specific_humidity(saturation_vapor_pressure,p):
    # Returns in kg / kg

    # specific humidity: q = mv / (mv + md) 
    # mass mixing ratio: w = mv / md
    # https://vortex.plymouth.edu/~stmiller/stmiller_content/Publications/AtmosRH_Equations_Rev.pdf eq 4
    # qs = es * ratio_of_molar_masses / (P - (1 - ratio_of_molar_mass)* es)
    m_dry = 28.9644 # more signficant digits
    m_h2o =  18.01528
    ratio_of_molar_masses = m_h2o / m_dry  # Approx 0.622
    denom = p - (1.0 - ratio_of_molar_masses) * saturation_vapor_pressure
    denom[denom < 0.01] = 0.01  # Avoiding negative numbers. 
                                # For the purpose of cloud_fraction
                                # this should be OK
    qs = ratio_of_molar_masses * saturation_vapor_pressure / denom
    return qs

def gfs_cloud_fraction(RH, saturation_specific_humidity,cloud_condensate):
    # https://dtcenter.ucar.edu/gmtb/users/ccpp/docs/sci_doc/group__module__radiation__clouds.html
    if False:
        print(f"min qs = {np.min(saturation_specific_humidity)}")
        print(f"min rh = {np.min(RH)}")
        print(f"max rh = {np.max(RH)}")
    ql = cloud_condensate * 100.0
    arg = (1.0 - RH) * saturation_specific_humidity
    # In case relative humidity is greater than 1.0
    arg[arg < 1.0e-06] = 1.0e-06

    denom = arg ** 0.49
    C = RH ** 0.25 * (1.0 - np.exp(-ql / denom))
    C[C>1.0] = 1.0
    return C

def replace_cloud_fraction(mode,month,year, base_directory):
    if False:
        # Comparing two methods of computing saturation vapor pressure
        freezing_kelvin = 273.1
        t = np.arange(-40, 60, 10)
        es1 = saturation_vapor_pressure(t + freezing_kelvin)
        es2 = saturation_vapor_pressure_2(t + freezing_kelvin) # more accurate
        for i,T in enumerate(t):
            print(f'temp = {T}, es1 = {es1[i]}, es2 = {es2[i]}, diff = {es1[i] - es2[i]} diff fraction = {(es1[i] - es2[i]) / es2[i]}')
    if True:
        file_name_input = f'{base_directory}{mode}/{year}/{month}/lw_input-{mode}-{year}-{month}.nc'
        file_name_new = f'{base_directory}{mode}/{year}/{month}/lw_input_mcica-{mode}-{year}-{month}.nc'

        cmd = f"cp -f {file_name_input} {file_name_new}"
        os.system(cmd)

        dt_input = Dataset(file_name_new, "a")
        temp_layer = dt_input["temp_layer"][:,:,:].data
        shape = temp_layer.shape
        temp_layer = temp_layer.reshape((shape[0]*shape[1],shape[2]))

        pres_layer = dt_input["pres_layer"][:,:,:].data
        shape = pres_layer.shape
        pres_layer = pres_layer.reshape((shape[0]*shape[1],shape[2])) 

        clwc = dt_input['q_liquid'][:,:].data
        ciwc = dt_input['q_ice'][:,:].data

        es = saturation_vapor_pressure_2(temp_layer)

        # mass mixing ratio
        w = dt_input['q'][:,:].data

        #print(f"min mixing ratio = {np.min(w)}")
        w[w<0] = 0.0

        rh = relative_humidity(w, es, pres_layer)
        qs = saturation_specific_humidity(es,pres_layer)
        C = gfs_cloud_fraction(rh, qs, clwc + ciwc)
        dt_input['cloud_fraction'][:] = C[:]

        dt_input.close()
        if False:
            shape = C.shape
            C = C.reshape((shape[0]*shape[1],))
            n = shape[0]*shape[1]
            rank = np.argsort(C)
            sorted_C = C[rank]

            print(f'min C = {np.min(C)}')
            print(f'mean C = {np.mean(C)}')
            print(f'max C = {np.max(C)}')

            print(f"C[0] = {sorted_C[0]}")
            print(f'C[10%] = {sorted_C[n // 10]}')
            print(f'C[20%] = {sorted_C[n * 2 // 10]}')
            print(f'C[40%] = {sorted_C[n * 4 // 10]}')
            print(f'C[60%] = {sorted_C[n * 6 // 10]}')
            print(f'C[90%] = {sorted_C[n * 9 // 10]}')
            print(f'C[95%] = {sorted_C[n * 95 // 100]}')
            print(f'C[98%] = {sorted_C[n * 98 // 100]}')
            print(f'C[99%] = {sorted_C[n * 99 // 100]}')
            print(f'C[99.9%] = {sorted_C[n * 999 // 1000]}')
            print(f'C[100%] = {sorted_C[-1]}')
            
def wrangle_zenodo():
    radiation = 'shortwave'
    abbv = 'sw'
    if False:
        mode = 'cross_validation'
        new_mode = 'validation'
        year = '2008'
        
    elif False:
        mode = 'testing'
        new_mode = 'testing'
        year = '2020'
        
    else:
        mode = 'training'
        new_mode = 'training'
        year = '2008'
        
    base_directory = f'/data-T1/hws/CAMS/processed_data/{mode}/{year}/'
    months = [str(m).zfill(2) for m in range(1,13)]
    input_prefix = f'nn_input_{abbv}-{mode}-{year}-' #06.nc
    output_prefix = f'{radiation}-{new_mode}-{year}-' #06.nc
    
    for month in months:
        file_name_input = f'{base_directory}{input_prefix}{month}.nc'
        file_name_output = f'{base_directory}{output_prefix}{month}.nc'
        cmd = f'cp {file_name_input} {file_name_output}'
        os.system(cmd)
        
    cmd = f'tar cvf {base_directory}{radiation}-{new_mode}-{year}.tar {base_directory}{output_prefix}*'
    os.system(cmd)
    
    cmd = f'zip {base_directory}{radiation}-{new_mode}-{year}.tar.zip {base_directory}{radiation}-{new_mode}-{year}.tar' 
    os.system(cmd)
    
def wrangle_zenodo_ukkonen():
    radiation = 'shortwave'
    abbv = 'sw'
    if False:
        mode = 'cross_validation'
        new_mode = 'validation'
        year = '2008'
        
    elif True:
        mode = 'testing'
        new_mode = 'testing'
        year = '2009'
        
    else:
        mode = 'training'
        new_mode = 'training'
        year = '2008'
        
    base_directory = f'/data-T1/hws/CAMS/processed_data/{mode}/{year}/'
    months = [str(m).zfill(2) for m in range(1,13)]
    input_prefix = f'Flux_Ukkonen-{year}-' #06.nc
    output_prefix = f'{radiation}-{new_mode}-{year}-' #06.nc
    
    for month in months:
        file_name_input = f'{base_directory}{input_prefix}{month}.nc'
        file_name_output = f'{base_directory}{output_prefix}{month}-ukkonen_format.nc'
        cmd = f'cp {file_name_input} {file_name_output}'
        os.system(cmd)
        
    cmd = f'tar cvf {base_directory}{radiation}-{new_mode}-{year}-ukkonen_format.nc.tar {base_directory}{output_prefix}*-ukkonen_format.nc'
    os.system(cmd)
    
    cmd = f'zip {base_directory}{radiation}-{new_mode}-{year}-ukkonen_format.nc.tar.zip {base_directory}{radiation}-{new_mode}-{year}-ukkonen_format.nc.tar' 
    os.system(cmd)
    
def examine_flux():
    mode = 'testing'
    new_mode = 'testing'
    year = '2009'
    month = '02'
    
    base_directory = f'/data-T1/hws/CAMS/processed_data/{mode}/{year}/'
    
    file_name_openbox = f'{base_directory}shortwave-{mode}-{year}-{month}.nc'
    dt_openbox = Dataset(file_name_openbox, "r")
    print (f"Opening: {file_name_openbox}")
    flux_down_direct = dt_openbox.variables['flux_down_direct'][:,:].data
    
    file_name_ukk = f'{base_directory}Flux_Ukkonen-{year}-{month}.nc'
    dt_ukk = Dataset(file_name_ukk, "r")
    rsd_ukk = dt_ukk.variables['rsd_dir'][:,:,:].data
    shape = rsd_ukk.shape
    rsd = rsd_ukk.reshape((shape[0] * shape[1], shape[2]))
    
    dt_openbox.close()
    dt_ukk.close()
    
    print(f'max rsd = {np.max(rsd)}')
    print(f'max flux down direct = {np.max(flux_down_direct)}')
    
    print(f'sum of difference = {np.sum(rsd - flux_down_direct)}')
    print(f'max of difference = {np.max(rsd - flux_down_direct)}')
    
    
    

if __name__ == "__main__":

    base_directory = f'/data-T1/hws/CAMS/processed_data/'
    #add_random_seed("testing", "06", "2009", base_directory)
    #compute_ecrad_output_data("testing", "06", "2009", base_directory, is_mcica=True, is_tmp=False, is_tripleclouds=False)
    
    if True:

        months = [str(m).zfill(2) for m in range(1,13)]
        
        modes = ('testing','validation','training')
        for mode in modes:
            if mode == 'testing':
                years = ['2009','2015','2020',]
            else:
                years = ['2008',]
            for year in years:

                for month in months:
                    
                    wrangle_sw_nn_input_data(mode, month, year, base_directory, is_mcica=False, is_tripleclouds=True)  
                    print(f"Completed {year} {month} {mode}", flush=True)
                    if False:
                        wrangle_openbox_to_ukkonen_input_data(
                            mode = mode,
                            month = month,
                            year = year,
                            base_directory = '/data-T1/hws/CAMS/processed_data/')
    
    if False:
        compare_ukkonen_input_data(
            mode = 'testing',
            month = '01',
            year = '2009',
            base_directory = '/data-T1/hws/CAMS/processed_data/')
        
    if False:
        wrangle_zenodo_ukkonen()
        #wrangle_zenodo()
        #examine_flux()

    if False:
        mode = 'training'
        month = '02'
        year = '2008'

        #wrapper_raw_sources (mode,month,year, base_directory)
        #wrangle_lw_nn_input_data(mode,month,year, base_directory)
        wrangle_sw_nn_input_data(mode, month, year, base_directory, is_mcica=True)  
        #examine_nn_input_data(mode,month,year, base_directory)

    if False:
        months = [str(m).zfill(2) for m in range(1,13)]
        combo = [('training','2008'),('cross_validation','2008'),('testing','2009'),('testing','2015'),('testing','2020'),]

        combo = [('validation','2008'),]
        #months = [str(m).zfill(2) for m in range(6,7)]

        for c in combo:
            mode = c[0]
            year = c[1]
            print(f'Processing {mode} {year}')
            for month in months[:]:
                print(f'{year} {month}')
                #wrangle_ecrad_input_data(mode, month, year, base_directory)
                #transform_ecrad_input_data(mode, month, year, base_directory,is_just_o2=True, is_mcica=True)
                compute_ecrad_output_data(mode, month, year, base_directory, is_mcica=True, is_tmp=False, is_tripleclouds=True)
                #wrangle_lw_nn_input_data(mode, month, year, base_directory)
                #wrangle_sw_nn_input_data(mode, month, year, base_directory, is_mcica=True)

                #transform_rte_rrtmgp_input_data(mode, month, year, base_directory)



                #wrapper_raw_sources (mode,month,year, base_directory)

                #compare_ecrad_with_rte_rrtmgp(mode,month,year, base_directory)
                #compare_ecrad_with_new_radiation(mode,month,year, base_directory, is_tmp=True)

                #compare_rte_rrtmgp_with_new_radiation(mode,month,year, base_directory)
                     


                #wrangle_sw_ukkonen_input_data(mode, month, year, base_directory)
                #replace_cloud_fraction(mode, month, year, base_directory)
    if False:
        examine_planck_2()


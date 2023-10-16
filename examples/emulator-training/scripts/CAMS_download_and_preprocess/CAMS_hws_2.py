#from netCDF4 import Dataset
import numpy as np
import cdsapi
import yaml
import time
import os

timestr = ['03:00', '09:00',  '15:00',   '21:00']
stepstr = ['3','9','15','21']
#timestr = [ '09:00',   '21:00']
#stepstr = ['9','21']
data_directory = "/data-T1/hws/CAMS/original_data/"

def get_dict_eac4_sfc(year, month, time):
    mydict = {
        # 'grid': [
        #     '15.0/30.0', # lat-lon 
        # ],
        'format': 'grib', #netcdf',  #'grib' 
        'variable': [
            'surface_pressure',
        ],
        'time': time,
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

def get_dict_eac4_ml(year,month, time):
    mydict =     {
                 'format': 'grib', #'netcdf',  #'grib' #'netcdf'
        'model_level': [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', 
            '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', 
            '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
        ],
        'time': time,
        # 'grid': [
        #     '15.0/30.0', # lat-lon 
        # ],
        'variable': [
            'nitrogen_dioxide', 'ozone', 
            'carbon_monoxide', 'specific_humidity','temperature', 
            'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
        ],
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

def get_dict_egg4_sfc(year,month, time):
    mydict =     {
        # 'grid': [
        #     '15.0/30.0', # lat-lon 
        # ],
         'format': 'grib', #'netcdf', 
          'variable': [
            '2m_temperature', 'forecast_albedo', 'toa_incident_solar_radiation',
        ],
        'step': time,
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

def get_dict_egg4_ml(year,month, time):
    mydict =     {
        # 'grid': [
        #     '15.0/30.0', # lat-lon 
        # ],
        'format': 'grib', #'netcdf', 
        'model_level': [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', 
            '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', 
            '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
        ],
         'variable': [
            'methane','carbon_dioxide',
        ],
        'step': time,
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

def get_cams_data():

    year = "2015"

    c = cdsapi.Client()

    with open('/home/hws/ADS/.cdsapirc', 'r') as f:
                credentials = yaml.safe_load(f)

    c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

    for month in ["02", "05", "08", "11"]:
        print(month)
        for i, t in enumerate([('03:00','3'), ('09:00','9'),  ('15:00', '15'),   ('21:00', '21')]):
            dict_eac4 = get_dict_eac4_ml(year,month,t[0])
            msg=c.retrieve(
            'cams-global-reanalysis-eac4', dict_eac4,
            data_directory +'CAMS_eac4_ml_%s%s%s01.grb'%(year,month,i))
            print(f"message = {msg}")


            dict_eac4 = get_dict_eac4_sfc(year,month,t[0])
            msg = c.retrieve(
                'cams-global-reanalysis-eac4', dict_eac4,
                data_directory +'CAMS_eac4_sfc_%s%s%s01.grb'%(year,month,i))
            print(f"message = {msg}")

                    
            # EGG4

            dict_egg4 = get_dict_egg4_ml(year,month,t[1])
            c.retrieve(
                'cams-global-ghg-reanalysis-egg4', dict_egg4,
                data_directory +'CAMS_egg4_ml_%s%s%s01.grb'%(year,month,i))
            
            dict_egg4 = get_dict_egg4_sfc(year,month,t[1])
            c.retrieve(
                'cams-global-ghg-reanalysis-egg4', dict_egg4,
                data_directory +'CAMS_egg4_sfc_%s%s%s01.grb'%(year,month,i))

def get_greenhouse_gas_inversion():
    year = "2015"

    c = cdsapi.Client()

    with open('/home/hws/ADS/.cdsapirc', 'r') as f:
                credentials = yaml.safe_load(f)

    c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

    file_name = data_directory + 'n2o/CAMS_n2o_%s.tar.old.gz'%(year)

    c.retrieve(
        'cams-global-greenhouse-gas-inversion',
        {
            'variable': 'nitrous_oxide',
            'quantity': 'concentration',
            'input_observations': 'surface',
            'time_aggregation': 'instantaneous',
            'version': 'v16r1', #'latest',
            'year': '%s'%(year),
            'month': [
                '02', #'05', '08',
                #'11',
            ],
            'format': 'tgz',
        },
        file_name)

def process_n2o(year):
    os.system("tar -xvzf {}".format(fname))
    os.system("conda activate netcdf3.8")
    os.system("cdo mergetime cams73_latest_n2o_conc_surface_inst_{}*.nc tmp.1.nc".format(year))
    os.system("ncrename -v level,mlev tmp.1.nc")
    os.system("ncrename -d level,mlev tmp.1.nc")
    os.system("ncrename -d hlevel,ilev tmp.1.nc")
    os.system("ncrename -d Psurf,aps tmp.1.nc")
    #assign_hybrid_coordinates()
    # Remove ap,bp and remap to the coarser grid
    os.system("ncks -O -x -v ap,bp tmp.1.nc tmp.2.nc")
    os.system("cdo remapbil,../icongrid_320km tmp.2.nc tmp.3.nc")

    # os.system("cdo remapbil,../newgrid tmp2.nc CAMS_n2o_{}_tmp.nc".format(year))
    # Now add the vertical reference
    os.system("ncks -A -v mlev,ilev,hyam,hybm,hyai,hybi VERTICAL_Ref.nc tmp.3.nc")
    os.system("ncrename -v Psurf,surface_air_pressure CAMS_n2o_{}_tmp.nc".format(year))
    os.system("rm tmp*")
    os.system("rm cams73*")
    os.system("rm *.tar.gz")

    # The vertical reference is inconsistent with the 3D variable
    # We need to flip the profiles so they are from top to bottom of atmosphere
    fname_tmp = "CAMS_n2o_{}_tmp.nc".format(year)
    dat = Dataset(dl_dir+fname_tmp,'a')
    sp_v = dat.variables['aps']
    sp_v.standard_name = "surface_air_pressure"
    n2o_dat = dat.variables['N2O'][:].data
    n2o_dat = np.flip(n2o_dat,axis=1)
    dat.variables['N2O'][:] = n2o_dat
    dat.close()

    # Almost done - now remap to the higher resolution vertical grid 
    # used by the main CAMS data
    fname_tmp2 = "CAMS_n2o_{}_tmp2.nc".format(year)
    fname_n2o = "CAMS_{}_n2o.nc".format(year)
    os.system("cdo remapeta,../newvct {} {}".format(fname_tmp,fname_tmp2))

    os.system("cdo -pressure_hl tmp.4.nc tmp.4.pressure.nc")
    os.system("ncrename -v lev,ilev tmp.4.pressure.nc")
    os.system("ncrename -d lev,ilev tmp.4.pressure.nc")
    os.system("ncks -A -v pressure tmp.4.pressure.nc tmp.4.nc")

    # Extract the time slices corresponding to the main CAMS data
    os.system ("ncks -d time,0,6,2 -d time,224,230,2 -d time,472,478,2 -d time,720,726,2 tmp.4.nc tmp.5.nc")

    # # FINALLY, concatenate N2O and main data files, write to final destination
    fname = "CAMS_{}.nc".format(year)

    os.system("ncks -A {} {}".format(fname_n2o,fname))
    os.system("ncatted -h -a history,global,d,, {}".format(fname))
    os.system("ncatted -h -a history_of_appended_files,global,d,, {}".format(fname))

    fname_final = "/media/peter/samsung/data/CAMS/CAMS_{}_2.nc".format(year)
    os.system("cp {} {}".format(fname,fname_final))


if __name__ == "__main__":
    st = time.perf_counter()
    #get_cams_data()
    get_greenhouse_gas_inversion()
    et = time.perf_counter()
    print(f"\nDownloaded one year of data in {et - st:0.4f} seconds")
#from netCDF4 import Dataset
import numpy as np
import cdsapi
import yaml
import time
import os
import datetime

timestr = ['03:00', '09:00',  '15:00',   '21:00']
stepstr = ['3','9','15','21']
#timestr = [ '09:00',   '21:00']
#stepstr = ['9','21']


def get_dict_eac4_sfc(year, month, day, hour):
    mydict = {
        'format': 'grib', 
        'variable': [
            'surface_pressure',
        ],
        'time': str(hour).zfill(2) + ':00',
        'date': f'{year}-{month}-{day}',
    }
    return mydict

def get_dict_eac4_ml(year, month, day, hour):
    mydict =     {
                 'format': 'grib', 
        'model_level': [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', 
            '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', 
            '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
        ],
        'time': str(hour).zfill(2) + ':00',
        'variable': [
            'nitrogen_dioxide', 'ozone', 
            'carbon_monoxide', 'specific_humidity','temperature', 
            'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
        ],
        'date': f'{year}-{month}-{day}',
    }
    return mydict

def get_dict_egg4_sfc(year, month, day, hour):
    mydict =     {
         'format': 'grib', 
          'variable': [
            '2m_temperature', 'forecast_albedo', 'toa_incident_solar_radiation',
        ],
        'step': str(hour),
        'date': f'{year}-{month}-{day}',
    }
    return mydict

def get_dict_egg4_ml(year, month, day, hour):
    mydict =     {
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
        'step': str(hour),
        'date': f'{year}-{month}-{day}',
    }
    return mydict

def get_cams_data(year,months,days,hours):

    #year = "2015"

    c = cdsapi.Client()

    with open('/home/hws/ADS/.cdsapirc', 'r') as f:
                credentials = yaml.safe_load(f)

    c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

    for month in months: #["02", "05", "08", "11"]:
        print(month)
        for day in days:
            for i, t in enumerate([('03:00','3'), ('09:00','9'),  ('15:00', '15'),   ('21:00', '21')]):
                dict_eac4 = get_dict_eac4_ml(year,month,day,t[0])
                msg=c.retrieve(
                'cams-global-reanalysis-eac4', dict_eac4,
                data_directory +'CAMS_eac4_ml_%s%s%s01.grb'%(year,month,i))
                print(f"message = {msg}")


                dict_eac4 = get_dict_eac4_sfc(year,month,day,t[0])
                msg = c.retrieve(
                    'cams-global-reanalysis-eac4', dict_eac4,
                    data_directory +'CAMS_eac4_sfc_%s%s%s01.grb'%(year,month,i))
                print(f"message = {msg}")

                        
                # EGG4

                dict_egg4 = get_dict_egg4_ml(year,month,day,t[1])
                c.retrieve(
                    'cams-global-ghg-reanalysis-egg4', dict_egg4,
                    data_directory +'CAMS_egg4_ml_%s%s%s01.grb'%(year,month,i))
                
                dict_egg4 = get_dict_egg4_sfc(year,month,day,t[1])
                c.retrieve(
                    'cams-global-ghg-reanalysis-egg4', dict_egg4,
                    data_directory +'CAMS_egg4_sfc_%s%s%s01.grb'%(year,month,i))


def download_cams_data_from_date_list(directory,date_list,hours):

    c = cdsapi.Client()

    with open('/home/hws/ADS/.cdsapirc', 'r') as f:
                credentials = yaml.safe_load(f)

    c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

    for i, d in enumerate(date_list):
        year = d[0]
        month = d[1]
        day = d[2]
        data_directory = directory + year + '/' + month +'/'
        for hour in hours:
            s_hour = str(hour).zfill(2)
            st = time.perf_counter()
            dict_eac4 = get_dict_eac4_ml(year,month,day,hour)
            msg=c.retrieve(
            'cams-global-reanalysis-eac4', dict_eac4,
            data_directory + f'CAMS_eac4_ml_{year}-{month}-{day}-{s_hour}.grb')
            print(f"message = {msg}")

            dict_eac4 = get_dict_eac4_sfc(year,month,day,hour)
            msg = c.retrieve(
                'cams-global-reanalysis-eac4', dict_eac4,
                data_directory + f'CAMS_eac4_sfc_{year}-{month}-{day}-{s_hour}.grb')
            print(f"message = {msg}")
                    
            # EGG4

            dict_egg4 = get_dict_egg4_ml(year,month,day,hour)
            c.retrieve(
                'cams-global-ghg-reanalysis-egg4', dict_egg4,
                data_directory + f'CAMS_egg4_ml_{year}-{month}-{day}-{s_hour}.grb')
            
            dict_egg4 = get_dict_egg4_sfc(year,month,day,hour)
            c.retrieve(
                'cams-global-ghg-reanalysis-egg4', dict_egg4,
                data_directory + f'CAMS_egg4_sfc_{year}-{month}-{day}-{s_hour}.grb')

            print(f"Downloaded {i+1}: {year}-{month}-{day}-{s_hour}")
            et = time.perf_counter()
            print(f" Elapsed Time: {et - st:0.4f} seconds", flush=True)

def get_greenhouse_gas_inversion(directory,year):
    year = "2015"

    c = cdsapi.Client()

    with open('/home/hws/ADS/.cdsapirc', 'r') as f:
                credentials = yaml.safe_load(f)

    c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

    months = [str(m).zfill(2) for m in range(1,13)]
    for month in months:
        data_directory = directory + year + '/' + month +'/'
        file_name = data_directory + f'CAMS_n2o_{year}-{month}.tar.gz'

        c.retrieve(
            'cams-global-greenhouse-gas-inversion',
            {
                'variable': 'nitrous_oxide',
                'quantity': 'concentration',
                'input_observations': 'surface',
                'time_aggregation': 'instantaneous',
                'version': 'v16r1', #'latest',
                'year': f'{year}',
                'month': [month,
                ],
                'format': 'tgz',
            },
            file_name)

def download_year_CAMS(year):
    data_directory = "/data-T1/hws/CAMS/original_data/"
    date_list = []
    #timestr = ['03:00', '09:00',  '15:00',   '21:00']
    #stepstr = ['3','9','15','21']
    # Every third hour
    hours = [i for i in range(0,25,3)]
    # Every 4th day
    for day_num in range(1,366,4):
        d = datetime.datetime(year,1,1) + datetime.timedelta(day_num - 1)
        month = d.strftime("%m")
        day = d.strftime("%d")
        date_list.append((str(year),month,day))
    download_cams_data_from_date_list(data_directory,date_list,hours)
    

          
     

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
    fname_tmp2 = "CAMS_n2o_{}_tmp2which.nc".format(year)
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
    #get_greenhouse_gas_inversion()
    year = 2008
    download_year_CAMS(year)
    et = time.perf_counter()
    print(f"\nDownloaded one year of data in {et - st:0.4f} seconds", flush=True)
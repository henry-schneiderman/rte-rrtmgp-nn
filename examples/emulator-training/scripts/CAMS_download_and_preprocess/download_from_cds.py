import numpy as np
import cdsapi
import yaml
import time
import datetime

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
            'skin_temperature'
        ],
        'step': str(hour),
        'date': f'{year}-{month}-{day}',
    }
    return mydict

def get_dict_egg4_sfc_st(year, month, day, hour):
    mydict =     {
         'format': 'grib', 
          'variable': [
            'skin_temperature'
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


def download_cams(directory,date_list,hours):

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


# Downloads skin temperature only
def download_cams_st(directory,date_list,hours):

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

            dict_egg4 = get_dict_egg4_sfc_st(year,month,day,hour)
            c.retrieve(
                'cams-global-ghg-reanalysis-egg4', dict_egg4,
                data_directory + f'CAMS_egg4_sfc_st_{year}-{month}-{day}-{s_hour}.grb')

            print(f"Downloaded {i+1}: {year}-{month}-{day}-{s_hour}")
            et = time.perf_counter()
            print(f" Elapsed Time: {et - st:0.4f} seconds", flush=True)

def download_greenhouse_gas_inversion(directory,year):

    c = cdsapi.Client()

    with open('/home/hws/ADS/.cdsapirc', 'r') as f:
                credentials = yaml.safe_load(f)

    c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

    months = [str(m).zfill(2) for m in range(1,13)]
    for month in months:
        st = time.perf_counter()
        data_directory = directory + str(year) + '/' + month +'/'
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
        et = time.perf_counter()
        print(f" Elapsed Time: {et - st:0.4f} seconds", flush=True)

def download_cams_year(directory, year):

    date_list = []

    # Every third hour
    hours = [i for i in range(0,24,3)]
    # Every 4th day
    for day_num in range(1,366,4):
        d = datetime.datetime(year,1,1) + datetime.timedelta(day_num - 1)
        month = d.strftime("%m")
        day = d.strftime("%d")
        date_list.append((str(year),month,day))
    download_cams(directory,date_list,hours)

# Downloads skin temperature only
def download_cams_year_st(directory, year):

    date_list = []

    # Every third hour
    hours = [i for i in range(0,24,3)]
    # Every 4th day
    for day_num in range(1,366,4):
        d = datetime.datetime(year,1,1) + datetime.timedelta(day_num - 1)
        month = d.strftime("%m")
        day = d.strftime("%d")
        date_list.append((str(year),month,day))
    download_cams_st(directory,date_list,hours)

if __name__ == "__main__":
    directory = "/data-T1/hws/CAMS/original_data/"
    st = time.perf_counter()
    year = 2008
    download_cams_year(directory, year)
    download_greenhouse_gas_inversion(directory, year)
    #download_cams_year_st(directory, year)
    if False:
        month = '01'
        day = '02'
        hour = 3
        s_hour = '03'

        c = cdsapi.Client()

        with open('/home/hws/ADS/.cdsapirc', 'r') as f:
                    credentials = yaml.safe_load(f)

        c = cdsapi.Client(url=credentials['url'], key=credentials['key'])
        dict_egg4 = get_dict_egg4_sfc(year,month,day,hour)
        c.retrieve(
                    'cams-global-ghg-reanalysis-egg4', dict_egg4,
                    directory + f'CAMS_egg4_sfc_{year}-{month}-{day}-{s_hour}.grb')
    et = time.perf_counter()
    print(f"\nDownloaded one year of data in {et - st:0.4f} seconds", flush=True)
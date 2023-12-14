# use conda activate pytorch2.0
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
            'skin_temperature', 'snow_albedo', 'snow_depth',
        ],
        'step': str(hour),
        'date': f'{year}-{month}-{day}',
    }
    return mydict


def get_dict_era5_sfc(year, month, day, hour):

    mydict =     {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': [
            'near_ir_albedo_for_diffuse_radiation', 'near_ir_albedo_for_direct_radiation', 'uv_visible_albedo_for_diffuse_radiation',
            'uv_visible_albedo_for_direct_radiation',
        ],
        'year': year,
        'month': month,
        'day': [
            day,
        ],
        'time': [
            str(hour).zfill(2) + ':00',
        ],
    }
    return mydict

def get_dict_era5_sfc_z(year, month, day, hour):

    mydict =     {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': [
            'geopotential', 
        ],
        'year': year,
        'month': month,
        'day': [
            day,
        ],
        'time': [
            str(hour).zfill(2) + ':00',
        ],
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

    with open('/home/hws/ADS/.cdsapirc', 'r') as f:
                credentials = yaml.safe_load(f)

    c = cdsapi.Client(url=credentials['url'], key=credentials['key'], timeout=600,quiet=False,debug=True)

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


def download_era5(directory,date_list,hours,is_era5_z=False):

    c_era = cdsapi.Client()

    for i, d in enumerate(date_list):
        year = d[0]
        month = d[1]
        day = d[2]
        data_directory = directory + year + '/' + month +'/'
        for hour in hours:
            s_hour = str(hour).zfill(2)
            st = time.perf_counter()

            if not is_era5_z:
                dict_era5 = get_dict_era5_sfc(year, month, day, hour)
                c_era.retrieve(
                 'reanalysis-era5-single-levels',
                 dict_era5,
                 data_directory + f'era5_sfc_{year}-{month}-{day}-{s_hour}.grb')
            else:
                dict_era5 = get_dict_era5_sfc_z(year, month, day, hour)
                c_era.retrieve(
                 'reanalysis-era5-single-levels',
                 dict_era5,
                 data_directory + f'era5_sfc_z_{year}-{month}-{day}-{s_hour}.grb')

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
        file_name = data_directory + f'CAMS_n2o_{year}-{month}.tar.latest.gz'

        c.retrieve(
            'cams-global-greenhouse-gas-inversion',
            {
                'variable': 'nitrous_oxide',
                'quantity': 'concentration',
                'input_observations': 'surface',
                'time_aggregation': 'instantaneous',
                'version': 'latest', #'v20r1', #v16r1', #'latest',
                'year': f'{year}',
                'month': [month,
                ],
                'format': 'tgz',
            },
            file_name)
        et = time.perf_counter()
        print(f" Elapsed Time: {et - st:0.4f} seconds", flush=True)

def day_list_to_date_list(day_list,year):
    date_list = []
    for day_num in day_list:
        d = datetime.datetime(year,1,1) + datetime.timedelta(day_num - 1)
        month = d.strftime("%m")
        day = d.strftime("%d")
        date_list.append((str(year),month,day))
    return date_list

if __name__ == "__main__":
    directory = "/data-T1/hws/CAMS/original_data/"
    st = time.perf_counter()


    #download_cams_year(directory, year)
    #download_greenhouse_gas_inversion(directory, year=2008)

    #download_cams_year_cross_validation(directory + "cross_validation/", year=2008, day_start=3)

    #download_cams_year_cross_validation(directory + "testing/", year=2020, day_start=4)

    #download_greenhouse_gas_inversion(directory + "testing/", year=2020)

    #download_era5_year(directory + "training/", year=2008, is_era5_z=True)

    hours = [i for i in range(0,24,3)]

    training_days = [day_num for  day_num in range(1,366,4)]
    training_dates_2008 = day_list_to_date_list(training_days,2008)

    cross_validation_days = [day_num for day_num in range(3,366,28)]
    cross_validation_dates_2008 = day_list_to_date_list(cross_validation_days, 2008)

    testing_days = [day_num for day_num in range(4,366,28)]
    testing_days_2009 = day_list_to_date_list(testing_days,2009)
    testing_dates_2020 = day_list_to_date_list(testing_days,2020) 
    testing_dates_2015 = day_list_to_date_list(testing_days,2015) 

    #download_cams(directory + "training/",training_dates_2008,hours)
    #download_cams(directory + "cross_validation/",cross_validation_dates_2008,hours)
    #download_cams(directory + "testing/", testing_dates_2009, hours)
    #download_cams(directory + "testing/", testing_dates_2020, hours)
    download_cams(directory + "testing/", testing_dates_2015, hours)

    #download_greenhouse_gas_inversion(directory + "testing/", year=2008)
    #download_greenhouse_gas_inversion(directory + "testing/", year=2009)
    download_greenhouse_gas_inversion(directory + "testing/", year=2015)
    #download_greenhouse_gas_inversion(directory + "testing/", year=2020)

    #download_era5(directory + "training/",training_dates_2008,hours)
    #download_era5(directory + "cross_validation/",cross_validation_dates_2008,hours)
    #download_era5(directory + "testing/", testing_dates_2009, hours)
    download_era5(directory + "testing/", testing_dates_2015, hours)
    #download_era5(directory + "testing/", testing_dates_2020, hours)

    if False:
        month = '01'
        day = '02'
        hour = 3
        s_hour = '03'

        c = cdsapi.Client()

        # with open('/home/hws/ADS/.cdsapirc', 'r') as f:
        #credentials = yaml.safe_load(f)

        #c = cdsapi.Client(url=credentials['url'], key=credentials['key'])
        dict_egg4 = get_dict_era5_sfc(year,month,day,hour)
        c.retrieve(
                    'reanalysis-era5-single-levels', dict_egg4,
                    directory + f'CAMS_era5_sfc_{year}-{month}-{day}-{s_hour}.grb')
    et = time.perf_counter()
    print(f"\nDownloaded one year of data in {et - st:0.4f} seconds", flush=True)
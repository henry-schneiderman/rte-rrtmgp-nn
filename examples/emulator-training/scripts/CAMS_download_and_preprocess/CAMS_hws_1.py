
#from netCDF4 import Dataset
import numpy as np
import cdsapi
#import yaml
import time
import os

timestr = ['03:00', '09:00',  '15:00',   '21:00']
stepstr = ['3','9','15','21']
#timestr = [ '09:00',   '21:00']
#stepstr = ['9','21']
def get_dict_eac4_sfc(year, month):
    mydict = {
        # 'grid': [
        #     '15.0/30.0', # lat-lon 
        # ],
        'format': 'grib', #netcdf',  #'grib' 
        'variable': [
            'surface_pressure',
        ],
        'time': timestr,
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

def get_dict_eac4_ml(year,month):
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
        'time': timestr,
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

def get_dict_egg4_sfc(year,month):
    mydict =     {
        # 'grid': [
        #     '15.0/30.0', # lat-lon 
        # ],
         'format': 'grib', #'netcdf', 
          'variable': [
            '2m_temperature', 'forecast_albedo', 'toa_incident_solar_radiation',
        ],
        'step': stepstr,
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

def get_dict_egg4_ml(year,month):
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
        'step': stepstr,
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

def get_cams_data():
    data_directory = "/data-T1/hws/CAMS/original_data/"

    year = "2015"

    c = cdsapi.Client()

    if False:
        with open('/home/hws/ADS/.cdsapirc', 'r') as f:
                credentials = yaml.safe_load(f)

        c = cdsapi.Client(url=credentials['url'], key=credentials['key'])
    else:
        c = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api/v2", key="15609:37efdd6a-929d-4381-b35f-1de84b98c2de")


    for month in ["02", "05", "08", "11"]:
        print(month)


        dict_eac4 = get_dict_eac4_ml(year,month)
        msg=c.retrieve(
        'cams-global-reanalysis-eac4', dict_eac4,
        data_directory +'CAMS_eac4_ml_%s%s01.grb'%(year,month))
        print(f"message = {msg}")

        if False:
            dict_eac4 = get_dict_eac4_sfc(year,month)
            msg = c.retrieve(
                'cams-global-reanalysis-eac4', dict_eac4,
                data_directory +'CAMS_eac4_sfc_%s%s01.grb'%(year,month))
            print(f"message = {msg}")

                    
            # EGG4

            dict_egg4 = get_dict_egg4_ml(year,month)
            c.retrieve(
                'cams-global-ghg-reanalysis-egg4', dict_egg4,
                data_directory +'CAMS_egg4_ml_%s%s01.grb'%(year,month))
            
            dict_egg4 = get_dict_egg4_sfc(year,month)
            c.retrieve(
                'cams-global-ghg-reanalysis-egg4', dict_egg4,
                data_directory +'CAMS_egg4_sfc_%s%s01.grb'%(year,month))


def get_dict_eac4_ml_2(year,month,hour):
    mydict =     {
                 'format': 'grib', #netcdf',  #'grib' #'netcdf'
        'model_level': [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', 
            '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', 
            '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
        ],
        'time': [
            '%s'%(hour),
        ],
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

def get_cams_data_2():
    data_directory = "/data-T1/hws/CAMS/"

    year = "2015"

    c = cdsapi.Client()

    with open('/home/hws/ADS/.cdsapirc', 'r') as f:
            credentials = yaml.safe_load(f)

    c = cdsapi.Client(url=credentials['url'], key=credentials['key'])


    for month in ["08", "11"]: #["02", "05", "08", "11"]:
        print(month)
        for i, hour in enumerate(["03:00", "15:00"]):
            dict_eac4 = get_dict_eac4_ml_2(year,month,hour)
            
            msg=c.retrieve(
            'cams-global-reanalysis-eac4', dict_eac4,
            data_directory +'CAMS_eac4_ml_%s%s%s01.grb'%(year,month,i))
           
            print(f"message = {msg}")

if __name__ == "__main__":
    st = time.perf_counter()
    get_cams_data()
    et = time.perf_counter()
    print(f"\nDownloaded one year of data in {et - st:0.4f} seconds")
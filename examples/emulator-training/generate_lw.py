import os
ex = './allsky_lw_gendata'

blocksize = '8'

mode = 'testing'
input_dir = f'/data-T1/hws/CAMS/processed_data/{mode}/'

year = '2015'
months = [str(m).zfill(2) for m in range(1,13)]
lw_kdist = '../../rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc'
lw_clouds = '../../extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-lw.nc'

combo = [('training','2008'),('cross_validation','2008'),('testing','2009'),('testing','2015'),('testing','2020')]

for c in combo:
    mode = c[0]
    year = c[1]
    input_dir = f'/data-T1/hws/CAMS/processed_data/{mode}/'
    for month in months[:]:

        input_file = f'{input_dir}{year}/{month}/CAMS_{year}-{month}.final.2.nc'
        output_file = f'{input_dir}{year}/Flux_lw-{year}-{month}.2.nc'
        cmd = f'{ex} {blocksize} {input_file} {lw_kdist} {lw_clouds} {output_file}'
        print (cmd)
        os.system(cmd)


#./allsky_sw_gendata 8 /data-T1/hws/CAMS/processed_data/training/2008/01/CAMS_2008-01.final.nc ../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc ../../extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc /data-T1/hws/tmp/Flux_sw-2008-01.nc
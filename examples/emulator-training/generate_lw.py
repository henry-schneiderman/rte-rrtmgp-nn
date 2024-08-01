import os
#ex = './allsky_lw_gendata'
ex = '/home/hws/ecrad/bin/ecrad /home/hws/ecrad/practical/config.2.nam'

#blocksize = '8'

#mode = 'testing'
#input_dir = f'/data-T1/hws/CAMS/processed_data/{mode}/'

#year = '2015'
months = [str(m).zfill(2) for m in range(1,13)]
#lw_kdist = '../../rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc'
#lw_clouds = '../../extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-lw.nc'

combo = [('training','2008'),('cross_validation','2008'),('testing','2009'),('testing','2015'),('testing','2020')]


#combo = [('training','2008'),('cross_validation','2008'),('testing','2009'),('testing','2015')]


for c in combo:
    mode = c[0]
    year = c[1]
    input_dir = f'/data-T1/hws/CAMS/processed_data/{mode}/'
    print(f'Processing {mode} {year}')
    for month in months[:]:
        print(f'{month}')
        input_file = f'{input_dir}{year}/{month}/lw_input-{mode}-{year}-{month}.nc'
        output_file = f'{input_dir}{year}/Flux_lw-{mode}-{year}-{month}.nc'
        #cmd = f'{ex} {blocksize} {input_file} {lw_kdist} {lw_clouds} {output_file}'
        cmd = f'{ex} {input_file} {output_file}'
        print (cmd)
        os.system(cmd)


#./allsky_sw_gendata 8 /data-T1/hws/CAMS/processed_data/training/2008/01/CAMS_2008-01.final.nc ../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc ../../extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc /data-T1/hws/tmp/Flux_sw-2008-01.nc
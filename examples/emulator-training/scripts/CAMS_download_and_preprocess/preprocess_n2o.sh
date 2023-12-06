# flip all coordinates: n2o, hyam, hyai, etc.
# remap vertical coord to 60 layers
# remap horizontal to icon grid

#python convert_to_hybrid_sigma_coordinates.py /data-T1/hws/CAMS/original_data/2008/01/n2o/tmp.1.nc
#ncks -O -x -v ap,bp /data-T1/hws/CAMS/original_data/2008/01/n2o/tmp.1.nc /data-T1/hws/CAMS/original_data/2008/01/n2o/tmp.2.nc
#ncrename -v surface_air_pressure,aps /data-T1/hws/CAMS/original_data/2008/01/n2o/tmp.2.nc
#cdo remapeta,/data-T1/hws/CAMS/newvct  /data-T1/hws/CAMS/original_data/2008/01/n2o/tmp.2.nc /data-T1/hws/CAMS/original_data/2008/01/n2o/tmp.3.nc
# cdo remapbil,/data-T1/hws/CAMS/icon_grid_0009_R02B03_R.nc /data-T1/hws/CAMS/original_data/2008/01/n2o/tmp.3.nc /data-T1/hws/CAMS/original_data/2008/01/n2o/tmp.4.nc
directory1='/data-T1/hws/CAMS/original_data/2008/01/'
directory2=${directory1}/n2o
fname=${directory1}'CAMS_n2o_2008-01.tar.gz'
year='2008'
month='01'
tar -xvzf ${fname} -C ${directory2}
conda activate netcdf3.8
cdo mergetime ${directory2}/cams73_latest_n2o_conc_surface_inst_{$year}{$month}.nc ${directory2}/tmp.1.nc
ncrename -v level,mlev tmp.1.nc
ncrename -d level,mlev tmp.1.nc
ncrename -d hlevel,ilev tmp.1.nc
ncrename -d Psurf,aps tmp.1.nc
    #assign_hybrid_coordinates()
    # Remove ap,bp and remap to the coarser grid
    ncks -O -x -v ap,bp tmp.1.nc tmp.2.nc
    cdo remapbil,../icongrid_320km tmp.2.nc tmp.3.nc

    # cdo remapbil,../newgrid tmp2.nc CAMS_n2o_{}_tmp.nc".format(year))
    # Now add the vertical reference
    ncks -A -v mlev,ilev,hyam,hybm,hyai,hybi VERTICAL_Ref.nc tmp.3.nc
    ncrename -v Psurf,surface_air_pressure CAMS_n2o_{}_tmp.nc".format(year)
    rm tmp*
    rm cams73*
    rm *.tar.gz

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
    cdo remapeta,../newvct {} {}".format(fname_tmp,fname_tmp2))

    cdo -pressure_hl tmp.4.nc tmp.4.pressure.nc")
    ncrename -v lev,ilev tmp.4.pressure.nc")
    ncrename -d lev,ilev tmp.4.pressure.nc")
    ncks -A -v pressure tmp.4.pressure.nc tmp.4.nc")

    # Extract the time slices corresponding to the main CAMS data
    os.system ("ncks -d time,0,6,2 -d time,224,230,2 -d time,472,478,2 -d time,720,726,2 tmp.4.nc tmp.5.nc")

    # # FINALLY, concatenate N2O and main data files, write to final destination
    fname = "CAMS_{}.nc".format(year)

    ncks -A {} {}".format(fname_n2o,fname))
    ncatted -h -a history,global,d,, {}".format(fname))
    ncatted -h -a history_of_appended_files,global,d,, {}".format(fname))

    fname_final = "/media/peter/samsung/data/CAMS/CAMS_{}_2.nc".format(year)
    cp {} {}".format(fname,fname_final))

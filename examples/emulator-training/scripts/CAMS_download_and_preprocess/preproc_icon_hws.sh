#!/bin/bash
echo "set conda environment:"
echo "> conda activate netcdf3.8"
echo "use 'source' instead of 'sh'"
echo "> source preproc_icon_hws.sh"
use_st=true
use_max_planck_grid=true

# See https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
#eval $(conda shell.bash hook)
conda activate netcdf3.8
if [ $# -ne 1 ] 
then
    echo "Usage: process_raw_cams_data.sh [year]"
else
    year=$1
    for month in '07' '08' '09' '10' '11' '12'
    do
        dir_0='/data-T1/hws/CAMS'
        dir_1='/data-T1/hws/CAMS/original_data/'${year}/${month}/
        dir_2='/data-T1/hws/CAMS/processed_data/'${year}/${month}/

        ## Merge into single file for each data set for each month

        # may need to be cdo -b F64
        cdo mergetime ${dir_1}/CAMS_eac4_ml_${year}-${month}-?????.grb ${dir_2}/CAMS_eac4_ml_${year}-${month}.grb 
        echo "Completed 1st mergetime"
        cdo mergetime ${dir_1}/CAMS_eac4_sfc_${year}-${month}-?????.grb ${dir_2}/CAMS_eac4_sfc_${year}-${month}.grb 
        cdo mergetime ${dir_1}/CAMS_egg4_ml_${year}-${month}-?????.grb ${dir_2}/CAMS_egg4_ml_${year}-${month}.grb 
        cdo mergetime ${dir_1}/CAMS_egg4_sfc_${year}-${month}-?????.grb ${dir_2}/CAMS_egg4_sfc_${year}-${month}.grb 
        echo "Completed standard mergetimes"

        # Remove the following when skin temperature is included 
        # in original downloaded CAMS_egg4_sfc (rather than downloaded separately)
        if [ "$use_st" = true ] 
        then
            cdo mergetime ${dir_1}/CAMS_egg4_sfc_st_${year}-${month}-?????.grb ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.grb
            cdo mergetime ${dir_1}/era5_sfc_${year}-${month}-?????.grb ${dir_2}/era5_sfc_${year}-${month}.grb
            echo "Completed st mergetimes"
        fi
        echo "Completed all mergetimes"

        ## Convert to netCDF files
        cdo --eccodes -f nc copy ${dir_2}/CAMS_eac4_ml_${year}-${month}.grb   ${dir_2}/CAMS_eac4_ml_${year}-${month}.nc
        echo "Completed 1st conversion to netcdf"
        cdo --eccodes -f nc copy ${dir_2}/CAMS_eac4_sfc_${year}-${month}.grb ${dir_2}/CAMS_eac4_sfc_${year}-${month}.nc 
        cdo --eccodes -f nc copy ${dir_2}/CAMS_egg4_ml_${year}-${month}.grb   ${dir_2}/CAMS_egg4_ml_${year}-${month}.nc
        cdo --eccodes -f nc copy ${dir_2}/CAMS_egg4_sfc_${year}-${month}.grb ${dir_2}/CAMS_egg4_sfc_${year}-${month}.nc 
        echo "Completed all conversions to netcdf"

        # Remove the following when skin temperature is included 
        # in original downloaded CAMS_egg4_sfc (rather than downloaded separately)
        if [ "$use_st" = true ] 
        then
            cdo --eccodes -f nc copy ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.grb ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.nc 
            cdo --eccodes -f nc copy ${dir_2}/era5_sfc_${year}-${month}.grb ${dir_2}/era5_sfc_${year}-${month}.nc 
        fi

        ## Reduce spatial resolution before attempting other operations

        if [ "$use_max_planck_grid" ] 
        then
            cdo remapcon,${dir_0}/icon_grid_0009_R02B03_R.nc ${dir_2}/CAMS_eac4_ml_${year}-${month}.nc ${dir_2}/CAMS_eac4_ml_${year}-${month}.icon.nc
            echo "Completed first spatial resolution remap"

            cdo remapcon,${dir_0}/icon_grid_0009_R02B03_R.nc ${dir_2}/CAMS_eac4_sfc_${year}-${month}.nc ${dir_2}/CAMS_eac4_sfc_${year}-${month}.icon.nc

            cdo remapcon,${dir_0}/icon_grid_0009_R02B03_R.nc ${dir_2}/CAMS_egg4_ml_${year}-${month}.nc ${dir_2}/CAMS_egg4_ml_${year}-${month}.icon.nc

            cdo remapcon,${dir_0}/icon_grid_0009_R02B03_R.nc ${dir_2}/CAMS_egg4_sfc_${year}-${month}.nc ${dir_2}/CAMS_egg4_sfc_${year}-${month}.icon.nc
            echo "Completed all spatial resolution remaps"

            if [ "$use_st" = true ] 
            then
                cdo remapcon,${dir_0}/icon_grid_0009_R02B03_R.nc ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.nc ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.icon.nc
                cdo remapcon,${dir_0}/icon_grid_0009_R02B03_R.nc ${dir_2}/era5_sfc_${year}-${month}.nc ${dir_2}/era5_sfc_${year}-${month}.icon.nc
            fi
        else
            cdo remapcon,${dir_0}/icongrid_320km ${dir_2}/CAMS_eac4_ml_${year}-${month}.nc ${dir_2}/CAMS_eac4_ml_${year}-${month}.icon.nc
            echo "Completed first spatial resolution remap"

            cdo remapcon,${dir_0}/icongrid_320km ${dir_2}/CAMS_eac4_sfc_${year}-${month}.nc ${dir_2}/CAMS_eac4_sfc_${year}-${month}.icon.nc

            cdo remapcon,${dir_0}/icongrid_320km ${dir_2}/CAMS_egg4_ml_${year}-${month}.nc ${dir_2}/CAMS_egg4_ml_${year}-${month}.icon.nc

            cdo remapcon,${dir_0}/icongrid_320km ${dir_2}/CAMS_egg4_sfc_${year}-${month}.nc ${dir_2}/CAMS_egg4_sfc_${year}-${month}.icon.nc
            echo "Completed all spatial resolution remaps"

            if [ "$use_st" = true ] 
            then
                cdo remapcon,${dir_0}/icongrid_320km ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.nc ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.icon.nc
                cdo remapcon,${dir_0}/icongrid_320km ${dir_2}/era5_sfc_${year}-${month}.nc ${dir_2}/era5_sfc_${year}-${month}.icon.nc
            fi
        fi

        ## Move relevant variables into CAMS_eac4_ml file
        ncks -A ${dir_2}/CAMS_eac4_sfc_${year}-${month}.icon.nc ${dir_2}/CAMS_eac4_ml_${year}-${month}.icon.nc
        ncks -A -v ch4,co2 ${dir_2}/CAMS_egg4_ml_${year}-${month}.icon.nc ${dir_2}/CAMS_eac4_ml_${year}-${month}.icon.nc
        # See https://codes.ecmwf.int/grib/param-db/
        ncrename -v \2t,t2m ${dir_2}/CAMS_egg4_sfc_${year}-${month}.icon.nc
        #ncrename -v var167,t2m ${dir_2}/CAMS_egg4_sfc_${year}-${month}.icon.nc
        #ncrename -v var243,fal ${dir_2}/CAMS_egg4_sfc_${year}-${month}.icon.nc
        #ncrename -v var212,tisr ${dir_2}/CAMS_egg4_sfc_${year}-${month}.icon.nc

        if [ "$use_st" = true ] 
        then
            #ncrename -v var235,skt ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.icon.nc
            #ncrename -v var32,asn ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.icon.nc
            #ncrename -v var141,sd ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.icon.nc
            ncks -A -v skt,asn,sd ${dir_2}/CAMS_egg4_sfc_st_${year}-${month}.icon.nc ${dir_2}/CAMS_eac4_ml_${year}-${month}.icon.nc
            #ncrename -v var18,alnid ${dir_2}/era5_sfc_${year}-${month}.icon.nc
            #ncrename -v var17,alnip ${dir_2}/era5_sfc_${year}-${month}.icon.nc
            #ncrename -v var16,aluvd ${dir_2}/era5_sfc_${year}-${month}.icon.nc
            #ncrename -v var15,aluvp ${dir_2}/era5_sfc_${year}-${month}.icon.nc
            ncks -A -v alnid,alnip,aluvd,aluvp ${dir_2}/era5_sfc_${year}-${month}.icon.nc ${dir_2}/CAMS_eac4_ml_${year}-${month}.icon.nc
            #ncks -A -v skin_temperature ${dir_2}/CAMS_egg4_sfc_${year}-${month}.icon.nc ${dir_2}/CAMS_eac4_ml_${year}-${month}.icon.nc
        fi

        ncks -A -v t2m,fal,tisr ${dir_2}/CAMS_egg4_sfc_${year}-${month}.icon.nc ${dir_2}/CAMS_eac4_ml_${year}-${month}.icon.nc

        cp ${dir_2}/CAMS_eac4_ml_${year}-${month}.icon.nc ${dir_2}/CAMS_${year}-${month}.nc

        # Generate pressures at interfaces between layers (i.e., at the 'levels')
        cdo -pressure_hl ${dir_2}/CAMS_${year}-${month}.nc ${dir_2}/CAMS_${year}-${month}.pressure_hl.nc
        cdo -pressure_fl ${dir_2}/CAMS_${year}-${month}.nc ${dir_2}/CAMS_${year}-${month}.pressure_fl.nc

        ncrename -d lev,level ${dir_2}/CAMS_${year}-${month}.pressure_hl.nc
        ncrename -d lev,layer ${dir_2}/CAMS_${year}-${month}.pressure_fl.nc
        ncrename -v pressure,pres_level ${dir_2}/CAMS_${year}-${month}.pressure_hl.nc
        ncrename -v pressure,pres_layer ${dir_2}/CAMS_${year}-${month}.pressure_fl.nc

        ncrename -d lev,layer ${dir_2}/CAMS_${year}-${month}.nc

        ncks -A -v pres_level ${dir_2}/CAMS_${year}-${month}.pressure_hl.nc ${dir_2}/CAMS_${year}-${month}.nc
        ncks -A -v pres_layer ${dir_2}/CAMS_${year}-${month}.pressure_fl.nc ${dir_2}/CAMS_${year}-${month}.nc

        cdo setcalendar,standard ${dir_2}/CAMS_${year}-${month}.nc ${dir_2}/CAMS_${year}-${month}.final.nc

        python add_solar_zenith.py ${dir_2}/CAMS_${year}-${month}.final.nc
    done
fi

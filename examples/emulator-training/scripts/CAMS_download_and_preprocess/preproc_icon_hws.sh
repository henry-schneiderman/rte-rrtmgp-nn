year=$1
for month in '01' '02'
do
    dir_1='/data-T1/hws/CAMS/original_data/'${year}
    dir_2='/data-T1/hws/CAMS/processed_data/'${year}
    # may need to be cdo -b F64
    cdo mergetime ${dir_1}/${month}/CAMS_eac4_ml_${year}-${month}-?????.grb ${dir_2}/${month}/CAMS_eac4_ml_${year}-${month}.grb 
    cdo mergetime ${dir_1}/${month}/CAMS_eac4_sfc_${year}-${month}-?????.grb ${dir_2}/${month}/CAMS_eac4_sfc_${year}-${month}.grb 
    cdo mergetime ${dir_1}/${month}/CAMS_egg4_ml_${year}-${month}-?????.grb ${dir_2}/${month}/CAMS_egg4_ml_${year}-${month}.grb 
    cdo mergetime ${dir_1}/${month}/CAMS_egg4_sfc_${year}-${month}-?????.grb ${dir_2}/${month}/CAMS_egg4_sfc_${year}-${month}.grb 
    # Remove the following when skin temperature is included 
    # in original downloaded CAMS_egg4_sfc (rather than downloaded separately)
    cdo mergetime ${dir_1}/${month}/CAMS_egg4_sfc_st_${year}-${month}-?????.grb ${dir_2}/${month}/CAMS_egg4_sfc_st_${year}-${month}.grb 

    cdo -f nc copy ${dir_2}/${month}/CAMS_eac4_ml_${year}-${month}.grb   ${dir_2}/${month}/CAMS_eac4_ml_${year}-${month}.nc
    cdo -f nc copy ${dir_2}/${month}/CAMS_eac4_sfc_${year}-${month}.grb ${dir_2}/${month}/CAMS_eac4_sfc_${year}-${month}.nc 
    cdo -f nc copy ${dir_2}/${month}/CAMS_egg4_ml_${year}-${month}.grb   ${dir_2}/${month}/CAMS_egg4_ml_${year}-${month}.nc
    cdo -f nc copy ${dir_2}/${month}/CAMS_egg4_sfc_${year}-${month}.grb ${dir_2}/${month}/CAMS_egg4_sfc_${year}-${month}.nc 
    # Remove the following when skin temperature is included 
    # in original downloaded CAMS_egg4_sfc (rather than downloaded separately)
    cdo -f nc copy ${dir_2}/${month}/CAMS_egg4_sfc_st_${year}-${month}.grb ${dir_2}/${month}/CAMS_egg4_sfc_st_${year}-${month}.nc 
done

ncks -A ${dir_2}/${month}/CAMS_eac4_sfc_${year}-${month}.nc ${dir_2}/${month}/CAMS_eac4_ml_${year}-${month}.nc
ncks -A -v ch4,co2 ${dir_2}/${month}/CAMS_egg4_ml_${year}-${month}.nc ${dir_2}/${month}/CAMS_eac4_ml_${year}-${month}.nc
# See https://codes.ecmwf.int/grib/param-db/
ncrename -v var167,t2m ${dir_2}/${month}/CAMS_egg4_sfc_${year}-${month}.nc
ncrename -v var243,fal ${dir_2}/${month}/CAMS_egg4_sfc_${year}-${month}.nc
ncrename -v var212,tisr ${dir_2}/${month}/CAMS_egg4_sfc_${year}-${month}.nc
# Add back next line and remove the following when skin temperature is included 
# in original downloaded CAMS_egg4_sfc (rather than downloaded separately)
#ncrename -v var235,skin_temperature ${dir_2}/${month}/CAMS_egg4_sfc_${year}-${month}.nc
ncrename -v var235,skin_temperature ${dir_2}/${month}/CAMS_egg4_sfc_st_${year}-${month}.nc

# Add back next line and remove the two following when skin temperature is included 
# in original downloaded CAMS_egg4_sfc (rather than downloaded separately)
#ncks -A -v t2m,fal,tisr,skin_temperature ${dir_2}/${month}/CAMS_egg4_sfc_${year}-${month}.nc ${dir_2}/${month}/CAMS_eac4_ml_${year}-${month}.nc
ncks -A -v t2m,fal,tisr ${dir_2}/${month}/CAMS_egg4_sfc_${year}-${month}.nc ${dir_2}/${month}/CAMS_eac4_ml_${year}-${month}.nc
ncks -A -v skin_temperature ${dir_2}/${month}/CAMS_egg4_sfc_st_${year}-${month}.nc ${dir_2}/${month}/CAMS_eac4_ml_${year}-${month}.nc

cdo remapcon,icongrid_320km ${dir_2}/${month}/CAMS_eac4_ml_${year}-${month}.nc ${dir_2}/${month}/CAMS_${year}-${month}.nc

# Generate pressures at interfaces between layers (i.e., at the 'levels')
cdo -pressure_hl ${dir_2}/${month}/CAMS_${year}-${month}.nc ${dir_2}/${month}/CAMS_${year}-${month}.pressure.nc

ncrename -d lev,layer ${dir_2}/${month}/CAMS_${year}-${month}.nc

ncks -x -v lev ${dir_2}/${month}/CAMS_${year}-${month}.nc ${dir_2}/${month}/CAMS_${year}-${month}.final.nc

ncks -A -v pressure ${dir_2}/${month}/CAMS_${year}-${month}.pressure.nc ${dir_2}/${month}/CAMS_${year}-${month}.final.nc
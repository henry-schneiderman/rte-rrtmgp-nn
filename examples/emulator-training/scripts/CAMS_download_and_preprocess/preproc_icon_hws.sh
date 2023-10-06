year=$1
dir_1='/data-T1/hws/CAMS/original_data'
dir_2='/data-T1/hws/CAMS/processed_data'
# may need to be cdo -b F64
cdo mergetime ${dir_1}/CAMS_eac4_ml_${year}?????.grb ${dir_2}/CAMS_eac4_ml_${year}.grb 
cdo mergetime ${dir_1}/CAMS_eac4_sfc_${year}?????.grb ${dir_2}/CAMS_eac4_sfc_${year}.grb 
cdo mergetime ${dir_1}/CAMS_egg4_ml_${year}?????.grb ${dir_2}/CAMS_egg4_ml_${year}.grb 
cdo mergetime ${dir_1}/CAMS_egg4_sfc_${year}?????.grb ${dir_2}/CAMS_egg4_sfc_${year}.grb 

cdo -f nc copy ${dir_2}/CAMS_eac4_ml_${year}.grb   ${dir_2}/CAMS_eac4_ml_${year}.nc
cdo -f nc copy ${dir_2}/CAMS_eac4_sfc_${year}.grb ${dir_2}/CAMS_eac4_sfc_${year}.nc 
cdo -f nc copy ${dir_2}/CAMS_egg4_ml_${year}.grb   ${dir_2}/CAMS_egg4_ml_${year}.nc
cdo -f nc copy ${dir_2}/CAMS_egg4_sfc_${year}.grb ${dir_2}/CAMS_egg4_sfc_${year}.nc 

ncks -A ${dir_2}/CAMS_eac4_sfc_${year}.nc ${dir_2}/CAMS_eac4_ml_${year}.nc
ncks -A -v ch4,co2 ${dir_2}/CAMS_egg4_ml_${year}.nc ${dir_2}/CAMS_eac4_ml_${year}.nc
# See https://codes.ecmwf.int/grib/param-db/
ncrename -v var167,t2m ${dir_2}/CAMS_egg4_sfc_${year}.nc
ncrename -v var243,fal ${dir_2}/CAMS_egg4_sfc_${year}.nc
ncrename -v var212,tisr ${dir_2}/CAMS_egg4_sfc_${year}.nc

ncks -A -v t2m,fal,tisr ${dir_2}/CAMS_egg4_sfc_${year}.nc ${dir_2}/CAMS_eac4_ml_${year}.nc

cdo remapcon,icongrid_320km ${dir_2}/CAMS_eac4_ml_${year}.nc ${dir_2}/CAMS_${year}.nc

# Generate pressures at interfaces between layers (i.e., at the 'levels')
cdo -pressure_hl ${dir_2}/CAMS_${year}.nc ${dir_2}/CAMS_${year}.pressure.nc

ncrename -d lev,layer ${dir_2}/CAMS_${year}.nc

ncks -x -v lev ${dir_2}/CAMS_${year}.nc ${dir_2}/CAMS_${year}.final.nc

ncks -A -v pressure ${dir_2}/CAMS_${year}.pressure.nc ${dir_2}/CAMS_${year}.final.nc
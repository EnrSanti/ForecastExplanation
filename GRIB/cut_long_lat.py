import xarray as xr

# Paths
def cut_grib_long_lat(grib_path,output_path, coordinates):

       ds = xr.open_dataset(grib_path, engine="cfgrib", decode_cf=True, decode_times=True, decode_timedelta=False)

       # Boolean mask for 2D curvilinear coordinates
       mask = (ds.longitude >= coordinates[0]) & (ds.longitude <= coordinates[1]) & \
              (ds.latitude >= coordinates[2]) & (ds.latitude <= coordinates[3])

       # Subset the dataset and drop points outside the mask
       ds_sub = ds.where(mask, drop=True)

       # Save to NetCDF
       ds_sub.to_netcdf(output_path)

       print(f"Subset dataset saved to {output_path}")
       ds.close()
       ds_sub.close()

import xarray as xr

grib_path = "./data/ECMWF-ITA_2018010100.grib"

# Specify variables one at a time for conflicting levels
variables_to_try = [
    {"typeOfLevel": "surface", "shortName": "msl"},
    {"typeOfLevel": "surface", "shortName": "z"},
    {"typeOfLevel": "surface", "shortName": "skt"},
    {"typeOfLevel": "isobaricInhPa"},  # Load compatible variables as a group
    {"typeOfLevel": "surface", "shortName": "sp"},
    {"typeOfLevel": "surface", "shortName": "sd"},
    {"typeOfLevel": "surface", "shortName": "cp"},
]

for filter_keys in variables_to_try:
    print(f"\n🔍 Trying filter: {filter_keys}")
    try:
        ds = xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs={'filter_by_keys': filter_keys})
        print("✅ Opened successfully.")
        print("📄 Dataset summary:\n", ds)
        print("\n📘 Variables:", list(ds.data_vars))
        print("🧭 Attributes:", ds.attrs)
    except Exception as e:
        print("❌ Failed with error:", e)

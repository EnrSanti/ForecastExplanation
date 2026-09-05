# Pipeline Data Flow

**This document has been written by AI model reading the code**

It describes the data files produced at each step of the pipeline, including their location, variables, and dimensions.
All heights `{h}` span the six pressure levels: `0300m`, `0500m`, `0700m`, `0850m`, `0925m`, `1000m`.

---

## Step 1 — Data Extraction

Reads the raw GRIB file for a given date, decodes it via cfgrib, cuts it to the configured region, and produces two
outputs.

### Input

- **Raw GRIB** — downloaded CERRA reanalysis file (e.g. `2009-01-02.grib`)

### Output 1 — Regional Cut

**Path:** `tmp_data/CERRA_cut/{date}/{date}_{region}_cut.nc`
**Dimensions:** `time × step × isobaricInhPa × y × x` (e.g. `8 × 3 × 6 × 76 × 63`)

| Variable | Description               |
|----------|---------------------------|
| `t`      | Temperature (K)           |
| `r`      | Relative humidity (%)     |
| `u`      | U-component of wind (m/s) |
| `v`      | V-component of wind (m/s) |
| `ccl`    | Cloud cover (%)           |

### Output 2 — Extracted Features

**Path:** `tmp_data/imgs_discrete/{date}/features.nc` (or `tmp_data/clustered/...`)
**Dimensions:** `time × y × x` (e.g. `24 × 76 × 63` — time/step flattened into 24 hourly frames)

| Variable                | Description                                         |
|-------------------------|-----------------------------------------------------|
| `temp_at_{h}`           | Normalized temperature [0–1] (for TOBAC)            |
| `humidity_at_{h}`       | Normalized humidity [0–1] (for TOBAC)               |
| `cloud_at_{h}`          | Normalized cloud cover [0–1] (for TOBAC)            |
| `raw_temp_at_{h}`       | Un-normalized temperature (K)                       |
| `raw_humidity_at_{h}`   | Un-normalized relative humidity (%)                 |
| `wind_at_{h}`           | Wind speed (m/s, not normalized)                    |
| `wind_direction_at_{h}` | Wind direction (degrees, meteorological convention) |

---

## Step 2 — Feature Detection & Tracking

Reads the extracted `features.nc`, runs TOBAC blob detection and tracking on the normalized fields, and produces three
separate output files.

### Input

- `tmp_data/imgs_discrete/{date}/features.nc` (from Step 1)

### Output 1 — Segmentation Masks

**Path:** `{run}/{date}/segmentation.nc`
**Dimensions:** `time × y × x` (e.g. `24 × 76 × 63`)
**Attributes:** `threshold` (e.g. `0.5`)

| Variable          | Description                                        |
|-------------------|----------------------------------------------------|
| `temp_at_{h}`     | Integer front/blob IDs from TOBAC (0 = background) |
| `humidity_at_{h}` | Integer front/blob IDs from TOBAC (0 = background) |
| `cloud_at_{h}`    | Integer front/blob IDs from TOBAC (0 = background) |

> Note: some height levels may be absent if TOBAC detected no valid features at that altitude.

### Output 2 — Raw Features

**Path:** `{run}/{date}/features.nc`
**Dimensions:** `time × y × x` (e.g. `24 × 76 × 63`)
**Attributes:** `dxy` — grid spacing in meters (e.g. `2500.0`)

| Variable                | Description                         |
|-------------------------|-------------------------------------|
| `raw_temp_at_{h}`       | Un-normalized temperature (K)       |
| `raw_humidity_at_{h}`   | Un-normalized relative humidity (%) |
| `wind_at_{h}`           | Wind speed (m/s)                    |
| `wind_direction_at_{h}` | Wind direction (degrees)            |

### Output 3 — Trajectories

**Path:** `{run}/{date}/trajectories.nc`
**Dimensions:** `index` (e.g. `487` tracked feature points)

| Variable    | Description                         |
|-------------|-------------------------------------|
| `cell`      | Cell/blob identifier                |
| `hdim_1`    | Grid index (y-axis)                 |
| `hdim_2`    | Grid index (x-axis)                 |
| `height`    | Height label (e.g. `temp_at_0300m`) |
| `latitude`  | Latitude of feature centroid        |
| `longitude` | Longitude of feature centroid       |
| `num`       | Feature number                      |
| `time`      | Timestamp                           |
| `time_cell` | Time since cell first appeared      |

---

## Step 3 — Reasoning

Reads both `segmentation.nc` and `features.nc` from Step 2, cross-references TOBAC blobs with physical values, and
produces tabular `.txt` reports.

### Input

- `{run}/{date}/segmentation.nc` (blob masks)
- `{run}/{date}/features.nc` (raw physical values + `dxy` attribute)

### Output — TSV Tables

**Path:** `{run}/{date}/reasoning/`

#### `winds.txt`

Mean wind within a 3 km radius of each city, per height and hour. Direction is the magnitude-weighted vector mean,
reported as a compass octave.

| Column           | Description                                 |
|------------------|---------------------------------------------|
| `timestamp`      | Hourly timestamp                            |
| `height`         | Pressure level                              |
| `lat`            | City latitude                               |
| `lon`            | City longitude                              |
| `wind_direction` | Compass octave (N, NE, E, SE, S, SW, W, NW) |
| `wind_speed`     | Mean wind speed (m/s)                       |

#### `cloud.txt`

Cloud segments detected by TOBAC that overlap each city. One row per cloud–city intersection, per height and hour.
Coverage is the percentage of the city's radius covered by the cloud.

| Column      | Description                                               |
|-------------|-----------------------------------------------------------|
| `timestamp` | Hourly timestamp                                          |
| `height`    | Pressure level                                            |
| `cloud_id`  | TOBAC blob ID                                             |
| `tot area`  | Total cloud segment area (km²)                            |
| `city`      | City name                                                 |
| `%covered`  | Percentage of the city's radius area covered by the cloud |

#### `heat.txt`

Mean temperature within a 3 km radius of each city, per height and hour.

| Column        | Description          |
|---------------|----------------------|
| `timestamp`   | Hourly timestamp     |
| `height`      | Pressure level       |
| `lat`         | City latitude        |
| `lon`         | City longitude       |
| `temperature` | Mean temperature (K) |

#### `heat_fronts.txt`

Temperature fronts detected by TOBAC, with their physical temperature and city membership.

| Column        | Description                                     |
|---------------|-------------------------------------------------|
| `timestamp`   | Hourly timestamp                                |
| `height`      | Pressure level                                  |
| `front_id`    | TOBAC blob ID                                   |
| `area`        | Front area (km²)                                |
| `cities`      | Comma-separated list of cities inside the front |
| `temperature` | Mean temperature of the front (K)               |

#### `humidity.txt`

Mean relative humidity within a 3 km radius of each city, per height and hour.

| Column      | Description                |
|-------------|----------------------------|
| `timestamp` | Hourly timestamp           |
| `height`    | Pressure level             |
| `lat`       | City latitude              |
| `lon`       | City longitude             |
| `humidity`  | Mean relative humidity (%) |

#### `humidity_fronts.txt`

Humidity fronts detected by TOBAC, with their physical humidity and city membership.

| Column      | Description                                     |
|-------------|-------------------------------------------------|
| `timestamp` | Hourly timestamp                                |
| `height`    | Pressure level                                  |
| `front_id`  | TOBAC blob ID                                   |
| `area`      | Front area (km²)                                |
| `cities`    | Comma-separated list of cities inside the front |
| `humidity`  | Mean relative humidity of the front (%)         |

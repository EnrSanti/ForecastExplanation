# Pipeline Data Flow

**This document has been written by an AI model reading the code**

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
| `front_at_{h}`          | Normalized theta-e gradient [0–1] (850/700/500 hPa) |
| `tadv_at_{h}`           | Normalized temperature advection [0–1], 0.5 = none  |
| `te_change_at_{h}`      | Normalized 3 h theta-e change [0–1], 0.5 = none     |
| `raw_front_at_{h}`      | Theta-e gradient (K/100 km)                         |
| `raw_tadv_at_{h}`       | Temperature advection (K/h)                         |
| `raw_te_change_at_{h}`  | 3 h theta-e change (K)                              |

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
| `front_at_{h}`    | Frontal zones (`front_at`, maximum), 850/700/500   |
| `warmadv_at_{h}`  | Warm advection (`tadv_at`, maximum)                |
| `tefall_at_{h}`   | Theta-e falling (`te_change_at`, minimum)          |
| `terise_at_{h}`   | Theta-e rising (`te_change_at`, maximum)           |

> Note: some height levels may be absent if TOBAC detected no valid features at that altitude.

### Output 2 — Raw Features

**Path:** `{run}/{date}/features.nc`
**Dimensions:** `time × y × x` (e.g. `24 × 76 × 63`)
**Attributes:** `dxy` — grid spacing in meters, from latitude/longitude (≈ `5490.0`)

| Variable                | Description                         |
|-------------------------|-------------------------------------|
| `raw_temp_at_{h}`       | Un-normalized temperature (K)       |
| `raw_humidity_at_{h}`   | Un-normalized relative humidity (%) |
| `raw_cloud_at_{h}`      | Cloud cover (%)                     |
| `wind_at_{h}`           | Wind speed (m/s)                    |
| `wind_direction_at_{h}` | Wind direction (degrees)            |
| `raw_front_at_{h}`      | Theta-e gradient (K/100 km)         |
| `raw_tadv_at_{h}`       | Temperature advection (K/h)         |
| `raw_te_change_at_{h}`  | 3 h theta-e change (K)              |

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
| `wind_direction` | Vector angle, clockwise rotation, 0 North   |
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

| Column                 | Description                                            |
|------------------------|---------------------------------------------------------|
| `timestamp`            | Hourly timestamp                                        |
| `height`               | Pressure level                                          |
| `front_id`             | TOBAC blob ID                                           |
| `area`                 | Front area (km²)                                        |
| `cities`               | Comma-separated list of cities inside the front         |
| `temperature_inside`   | Mean temperature inside the front (K)                   |
| `temperature_outside`  | Mean temperature outside all fronts, background only (K)|

#### `cloud_cover.txt`

Mean CERRA cloud cover within a 3 km radius of each city, per height and hour, whether or not TOBAC segmented a
cloud there.

| Column      | Description         |
|-------------|---------------------|
| `timestamp` | Hourly timestamp    |
| `height`    | Pressure level      |
| `lat`       | City latitude       |
| `lon`       | City longitude      |
| `cloud`     | Mean cloud cover (%) |

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

| Column               | Description                                                  |
|----------------------|---------------------------------------------------------------|
| `timestamp`          | Hourly timestamp                                              |
| `height`             | Pressure level                                                |
| `front_id`           | TOBAC blob ID                                                 |
| `area`               | Front area (km²)                                              |
| `cities`             | Comma-separated list of cities inside the front                |
| `humidity_inside`    | Mean relative humidity inside the front (%)                    |
| `humidity_outside`   | Mean relative humidity outside all fronts, background only (%) |

#### `front_fronts.txt`, `warmadv_fronts.txt`, `tefall_fronts.txt`, `terise_fronts.txt`

Same columns as `heat_fronts.txt`, for the front segments of Step 2; the values (`{name}_inside`, `{name}_outside`)
come from `raw_front`, `raw_tadv` and `raw_te_change`.

---

## Step 4 — Ground Truth Generation

Extracts the official weather forecast from ARPA FVG XML (or PDF fallback) files for the day before each target date,
producing a structured JSON with per-city forecast descriptions.

### Input

- XML forecast files in `./xmls/` (preferred), or PDF fallback

### Output

**Path:** `{run}/{date}/gt.json`

A JSON object keyed by date, mapping each city to its forecast fields (e.g. `PIOGGIA_DESCRIZIONE`,
`CIELO_DESCRIZIONE`).

---

## Step 5 — Translation

Reads the reasoning TSV files and the ground-truth JSON for each day and flattens them into a single CSV row per city,
suitable for downstream ML models. Currently the only translator is `FoldRmTranslator`.

### Input

- `{run}/{date}/gt.json` (from Step 4)
- `{run}/{date}/reasoning/winds.txt` (from Step 3)
- `{run}/{date}/reasoning/cloud.txt` (from Step 3)
- `{run}/{date}/reasoning/heat.txt` (from Step 3)
- `{run}/{date}/reasoning/humidity.txt` (from Step 3)
- `{run}/{date}/reasoning/cloud_cover.txt` (from Step 3)
- `{run}/{date}/reasoning/heat_fronts.txt` (from Step 3)
- `{run}/{date}/reasoning/humidity_fronts.txt` (from Step 3)
- `{run}/{date}/reasoning/{front,warmadv,tefall,terise}_fronts.txt` (from Step 3)

### Output

**Path:** `{run}/translated/{date}.csv`

One row per city. Per-hour, per-level values are grouped before being written out: hours are bucketed into
`early_morning` (00–06), `morning` (07–12), `afternoon` (13–18), `evening` (19–23), and pressure levels into
`low` (1000/0925/0850), `medium` (0700/0500), `high` (0300). Columns are structured as `{feature}_{time_group}_{height_group}`:

| Column group                                          | Description                                                                                                                   |
|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `prev_pioggia`                                        | Encoded rain forecast (`rain_enum`)                                                                                           |
| `prev_cloud`                                          | Encoded sky forecast (`cloud_enum`)                                                                                           |
| `month`                                               | Month of the target date                                                                                                      |
| `location`                                            | City slug                                                                                                                     |
| `wind_direction_*`                                    | Resultant vector wind direction, compass octave, per time/height group                                                        |
| `wind_speed_*`                                        | Resultant vector wind speed (m/s), per time/height group                                                                      |
| `coverage_clouds_*`                                   | Cloud coverage (%), averaged per time/height group                                                                            |
| `size_cloud_*`                                        | Cloud segment area (km²), averaged per time/height group                                                                      |
| `temperature_*`                                       | Temperature (K), averaged per time/height group                                                                               |
| `humidity_*`                                          | Relative humidity (%), averaged per time/height group                                                                         |
| `{f}_hours_{g}`                                       | Hours the city is inside a segment of `f` at a level of group `g`                                                             |
| `{f}_value_{g}`                                       | Mean value inside the segments containing the city (0 if none)                                                                |
| `region_{f}_area_{g}`                                 | Largest total segment area (km²) of an hour and level, same for every row of the day                                          |
| `cloud_cover_*`                                       | CERRA cloud cover (%), averaged per time/height group                                                                         |
| `cloud_total_*`                                       | Column cloud cover (%, hourly max over the levels), per time group and `day`                                                  |
| `rh925_max`, `rh850_max`, `rh700_max`                 | Max relative humidity (%) at 925/850/700 hPa                                                                                  |
| `sat850_hours`, `sat700_hours`                        | Hours with RH >= 90% at 850/700 hPa                                                                                           |
| `sat_column_hours`                                    | Hours with RH >= 90% at 925 and 850 hPa and >= 85% at 700 hPa                                                                 |
| `south850_mean`, `south850_max`, `south700_mean`      | Southerly wind component (m/s, wind from the south > 0)                                                                       |
| `moist_flux850_mean`, `moist_flux850_max`             | Southerly 850 hPa wind times its RH: moist inflow against the Alps                                                            |
| `lapse_850_500`                                       | Mean 850-500 hPa temperature difference (K), static stability                                                                 |
| `cloud_low_day`, `cloud_medium_day`, `cloud_high_day` | Daily mean cloud cover (%) per height group                                                                                   |
| `cloud_mid_max`, `cloud_mid_hours`                    | Max 700/500 hPa cloud cover (%) and hours with it >= 80%                                                                      |
| `cloud_850_max`                                       | Max 850 hPa cloud cover (%)                                                                                                   |
| `te850_mean`                                          | Mean 850 hPa theta-e (K)                                                                                                      |
| `te850_drop6h`, `te850_rise6h`, `t850_drop6h`         | Largest 6 h theta-e drop / rise and temperature drop at 850 hPa (K)                                                           |
| `instab_te850_500`                                    | Max theta-e 850 − 500 hPa (K), > 0 unstable                                                                                   |
| `region_*`                                            | `rh850`, `rh700`, `sat700_hours`, `south850`, `cloud_mid`, `te850_tend` pooled over all the cities, same for every row of the day |

`f` is one of `humidity_fronts` (RH < 60%), `temperature_fronts` (warmest areas), with `g` in low/medium/high, or
`front`, `warmadv`, `tefall`, `terise`, with `g` in low (850) / medium (700/500).

After all requested dates are translated, the per-day CSVs are concatenated into a single merged dataset at
`{run}/translated/{min_date}_{max_date}.csv`.


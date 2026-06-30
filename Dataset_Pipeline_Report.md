# Dataset Pipeline Report
## UK Road Safety — STATS19 Accident Data (2005–2017)

---

## 1. Raw Dataset Overview

- **Source:** STATS19 — official UK police accident reporting system
- **Shape:** 2,047,256 rows × 34 columns
- **Coverage:** Every road accident reported to police in Great Britain from 2005 to 2017

---

## 2. Column-by-Column Journey

### RAW → DROPPED (never used in model)

| Column | Raw Type | Why Dropped |
|--------|----------|-------------|
| `Accident_Index` | object | Administrative police reference number only. Encodes no accident information — using it would cause the model to memorise individual records (overfitting). |
| `1st_Road_Number` | int | Specific road number (e.g. A1, M25). Too granular and sparse — thousands of unique values with no ordinal meaning for severity prediction. |
| `2nd_Road_Number` | int | Same problem as `1st_Road_Number`. |
| `Did_Police_Officer_Attend_Scene_of_Accident` | object | Administrative field. Whether a police officer attended is a consequence of severity, not a cause — using it would cause data leakage. |
| `Local_Authority_(Highway)` | object | Highway authority code. Redundant with location already captured by `Urban_or_Rural_Area`, `InScotland`, `Latitude`, and `Longitude`. |
| `Location_Easting_OSGR` | float | Ordnance Survey grid easting — a different coordinate system redundant with `Latitude`/`Longitude`. |
| `Location_Northing_OSGR` | float | Ordnance Survey grid northing — same reason. |
| `LSOA_of_Accident_Location` | object | Lower Super Output Area — a UK census geography code. 35,000+ unique values, too high cardinality for direct use and fully redundant with coordinates. |
| `Local_Authority_(District)` | object | District council name — 400+ categories, high cardinality, redundant with coordinates. |
| `Police_Force` | object | Police force area — 52 categories, essentially a proxy for geography already covered by coordinates and `InScotland`. |
| `Special_Conditions_at_Site` | object | **97.4% null** — cannot be salvaged by any imputation method. Dropped. |
| `Carriageway_Hazards` | object | **98.1% null** — same situation. Dropped. |
| `Time` | object | Dropped after extracting `Hour` from it (see Feature Engineering). |
| `Date` | object | Dropped after extracting `Month` and `Season` from it. |
| `Day_of_Week` | object | Dropped after engineering `is_weekend` binary flag from it. |
| `Accident_Severity` | object | The original 3-class target (Slight / Serious / Fatal). Dropped after creating the binary `Severity` column (0/1). |
| `Latitude` | float | Kept through cleaning but dropped before modeling — used for EDA and maps only, not fed into the classifier. |
| `Longitude` | float | Same as Latitude. |
| `Number_of_Casualties` | int | Dropped just before feature selection. This is a consequence of the accident outcome, not a cause — using it would be target leakage (you cannot know casualty count before the accident happens). |

---

### RAW → CLEANED → KEPT IN MODEL

| Column | Raw State | Cleaning Applied | Final Form in Model |
|--------|-----------|-----------------|---------------------|
| `Year` | int64, clean | No changes needed | `Year` — integer as-is |
| `InScotland` | object: "Yes"/"No" | Converted to binary integer: Yes=1, No=0 | `InScotland` — int (0/1) |
| `Urban_or_Rural_Area` | object: Urban / Rural / Unallocated | Dropped 394 "Unallocated" rows (meaningless category). Remaining two categories ordinal-encoded. | **Ordinal encoded:** Urban=0, Rural=1 |
| `1st_Road_Class` | object: Motorway / A(M) / A / B / C / Unclassified | Clean, no nulls. Ordinal-encoded by road hierarchy (Unclassified=1 → Motorway=6). | **Ordinal encoded:** 1–6 |
| `2nd_Road_Class` | object: same categories + NaN when not at junction | NaN was a structural null (not missing data — means "no second road"). Filled with "Not at Junction" category, then ordinal-encoded. | **Ordinal encoded:** Not at Junction=0, Unclassified=1 → Motorway=6 |
| `Road_Type` | object: Single carriageway / Dual / Roundabout / One way / Slip road / Unknown | "Data missing or out of range" kept as "Unknown" category — rows have valid data in all other columns so dropping would waste information. Ordinal-encoded by severity risk. | **Ordinal encoded:** Unknown=0, Single carriageway=1, Slip road=2, One way street=3, Roundabout=4, Dual carriageway=5 |
| `Speed_limit` | float: 20/30/40/50/60/70 + 99 (unknown code) + 0/10/15 (non-standard) | 99.0 replaced with NaN. NaN imputed using **median grouped by Road_Type** (e.g. motorway NaN → 70). 36 rows with non-standard speeds (0, 10, 15 mph) dropped. | `Speed_limit` — int: 20/30/40/50/60/70 |
| `Junction_Detail` | object: 9 categories + "Data missing or out of range" | 733 "Data missing or out of range" rows dropped (too few to matter, no way to impute junction type). | **One-Hot encoded** (9 categories → 8 binary columns, drop_first=True) |
| `Junction_Control` | object: 6 categories + "Data missing or out of range" | "Data missing or out of range" relabelled to "Unknown" — kept as valid category. | **One-Hot encoded** (6 categories → 5 binary columns) |
| `Light_Conditions` | object: 5 categories + "Data missing or out of range" | 13 rows relabelled to "Unknown". Ordinal-encoded by darkness level (Daylight=0, full darkness=highest). | **Ordinal encoded:** Daylight=0 → Darkness: no lighting=4 |
| `Weather_Conditions` | object: 9 categories + two different "unknown" labels | Both unknown labels merged into single "Unknown" category. | **One-Hot encoded** (9 categories → 8 binary columns) |
| `Road_Surface_Conditions` | object: 5 categories + "Data missing or out of range" | Relabelled to "Unknown". Ordinal-encoded by slipperiness risk. | **Ordinal encoded:** Dry=0 → Flood=4 |
| `Pedestrian_Crossing-Human_Control` | float: 0.0 = none, other values = type of control | Nulls imputed with **mode grouped by Road_Type**. Then binarized: 0.0 → 0 (no control present), anything else → 1 (control present). | **Binary:** 0 or 1 |
| `Pedestrian_Crossing-Physical_Facilities` | float: same structure | Same treatment as Human_Control. | **Binary:** 0 or 1 |
| `Number_of_Vehicles` | int, clean | No changes needed through cleaning. Dropped just before feature selection (post-accident information). | Dropped before modeling |
| `Hour` | Did not exist in raw data | **Engineered** from `Time` column (HH:MM format → integer 0–23). NaNs filled with mode hour. | `Hour` — int: 0–23 |
| `Month` | Did not exist in raw data | **Engineered** from `Date` column (datetime → month integer 1–12). | `Month` — int: 1–12 |
| `Season` | Did not exist in raw data | **Engineered** from `Month`: Dec/Jan/Feb=Winter, Mar/Apr/May=Spring, Jun/Jul/Aug=Summer, Sep/Oct/Nov=Autumn. | **One-Hot encoded** (4 categories → 3 binary columns) |
| `is_rush_hour` | Did not exist | **Engineered** from `Hour`: 1 if 7–9am or 4–7pm, else 0. | Binary: 0 or 1 |
| `is_night` | Did not exist | **Engineered** from `Hour`: 1 if hour ≥ 20 or ≤ 6, else 0. | Binary: 0 or 1 |
| `is_weekend` | Did not exist | **Engineered** from `Day_of_Week`: 1 if Saturday or Sunday, else 0. | Binary: 0 or 1 |
| `Severity` | Did not exist | **Engineered** from `Accident_Severity`: 0 if Slight, 1 if Serious or Fatal (binary target). | **Target variable y** |

---

## 3. Row Changes Summary

| Stage | Rows Remaining | Rows Lost | Reason |
|-------|---------------|-----------|--------|
| Raw dataset | 2,047,256 | — | Starting point |
| After dropping duplicates | ~2,047,000 | ~256 | Exact duplicate rows removed |
| After dropping 0-casualty rows | ~2,046,900 | ~100 | STATS19 records only personal injury accidents — 0 casualties is a data error |
| After dropping null coordinates | ~2,046,899 | 1 | Single row with no Latitude/Longitude — unrecoverable |
| After dropping Junction_Detail unknowns | ~2,046,166 | 733 | "Data missing or out of range" — not imputeable |
| After dropping Urban/Rural "Unallocated" | ~2,046,100 | ~66 | Unallocated category has no meaning for modeling |
| After dropping non-standard speed limits | ~2,046,064 | 36 | Speeds of 0, 10, 15 mph are recording errors |
| **Final cleaned dataset** | **2,046,228** | **1,028 total** | **0 nulls remaining** |

---

## 4. Column Count Changes

| Stage | Columns |
|-------|---------|
| Raw dataset | 34 |
| After dropping 10 admin/redundant columns | 24 |
| After dropping 2 near-empty columns | 22 |
| After Feature Engineering (+5 new, -1 Time, -1 Date) | 26 |
| After Encoding (OHE expands nominal cols) | 42 |
| After dropping Number_of_Casualties | **41** |

---

## 5. Final Model-Ready Dataset

- **Shape:** 2,046,228 rows × 41 features
- **Target:** `Severity` — binary (0 = Slight: 84.7%, 1 = Serious/Fatal: 15.3%)
- **Nulls:** 0
- **Dtypes:** All integer or float — no strings, no categories

### Final 41 Features

**Continuous/Ordinal (kept as integers):**
`Year`, `InScotland`, `Urban_or_Rural_Area`, `1st_Road_Class`, `2nd_Road_Class`, `Road_Type`, `Speed_limit`, `Light_Conditions`, `Road_Surface_Conditions`, `Pedestrian_Crossing-Human_Control`, `Pedestrian_Crossing-Physical_Facilities`, `Hour`, `Month`, `is_rush_hour`, `is_night`, `is_weekend`

**One-Hot Encoded (binary columns):**
- `Weather_Conditions_*` — 8 columns
- `Junction_Detail_*` — 8 columns
- `Junction_Control_*` — 5 columns
- `Season_*` — 3 columns

**Train/Test Split:** 80/20 stratified
- Training: 1,636,982 rows
- Test: 409,246 rows
- Both sets preserve the original 84.7/15.3 class ratio exactly


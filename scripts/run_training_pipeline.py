from __future__ import annotations

import json
import math
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


warnings.filterwarnings("ignore", category=RuntimeWarning)


OUT_ROOT = Path(os.environ.get("TC_DETECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_DIR = OUT_ROOT / "data"
MODEL_DIR = OUT_ROOT / "models"
FIG_DIR = OUT_ROOT / "figures"
REPORT_DIR = OUT_ROOT / "reports"
SCRIPT_DIR = OUT_ROOT / "scripts"

IBTRACS = Path(os.environ.get("IBTRACS_PATH", r"D:\path\to\IBTrACS.ALL.v04r01.nc"))
ERA5_ALL = Path(os.environ.get("ERA5_ALL_PATH", r"E:\path\to\all_era5.nc"))
CESM_CTL_PRE = Path(os.environ.get("CESM_CTL_PRE", r"E:\path\to\CESM\control"))
CESM_GGW_PRE = Path(os.environ.get("CESM_GGW_PRE", r"G:\path\to\CESM\surface"))
OWZ_CTL = Path(os.environ.get("OWZ_CTL", r"E:\path\to\OWZ\control"))
OWZ_GGW = Path(os.environ.get("OWZ_GGW", r"G:\path\to\OWZ\surface"))

RNG = np.random.default_rng(404)
EARTH_R_KM = 6371.0
EARTH_R_M = 6371000.0
LEVELS = [200, 300, 500, 700, 850]
FEATURE_COLUMNS = [
    "lat",
    "lon_sin",
    "lon_cos",
    "month_sin",
    "month_cos",
    "u_200",
    "v_200",
    "t_200",
    "q_200",
    "rh_200",
    "u_300",
    "v_300",
    "t_300",
    "q_300",
    "rh_300",
    "u_500",
    "v_500",
    "t_500",
    "q_500",
    "rh_500",
    "u_700",
    "v_700",
    "t_700",
    "q_700",
    "rh_700",
    "u_850",
    "v_850",
    "t_850",
    "q_850",
    "rh_850",
    "zeta_850",
    "zeta_500",
    "wshear_850_200",
    "steer_u_850_200",
    "steer_v_850_200",
    "zeta850_max_r500",
    "zeta500_max_r500",
    "rh700_mean_r500",
    "shear_min_r500",
]


@dataclass
class SplitYears:
    train_end: int = 2010
    val_end: int = 2015


def ensure_dirs() -> None:
    for path in [DATA_DIR, MODEL_DIR, FIG_DIR, REPORT_DIR, SCRIPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def copy_this_script() -> None:
    src = Path(__file__).resolve()
    dst = SCRIPT_DIR / src.name
    if src != dst:
        shutil.copy2(src, dst)


def lon360(lon: float | np.ndarray) -> float | np.ndarray:
    return np.mod(lon, 360.0)


def lon_diff_deg(lon2: np.ndarray | float, lon1: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(lon2) - np.asarray(lon1) + 180.0) % 360.0 - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon_diff_deg(lon2, lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return EARTH_R_KM * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def dxdy_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = lon_diff_deg(lon2, lon1) * 111.32 * np.cos(np.deg2rad(lat1))
    return dx, dy


def decode_bytes_array(arr: np.ndarray) -> np.ndarray:
    return arr.astype(str)


def rh_from_q_t_p(q: np.ndarray, t: np.ndarray, p_hpa: float) -> np.ndarray:
    tc = t - 273.15
    es = 6.112 * np.exp((17.67 * tc) / (tc + 243.5))
    eps = 0.622
    qs = eps * es / np.maximum(p_hpa - (1.0 - eps) * es, 1.0)
    return np.clip(100.0 * q / qs, 0.0, 150.0)


def compute_zeta(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    dlat = np.deg2rad(abs(float(lats[1] - lats[0])))
    dlon = np.deg2rad(abs(float(lons[1] - lons[0])))
    dy = EARTH_R_M * dlat
    dx = EARTH_R_M * np.cos(np.deg2rad(lats)) * dlon
    dx = np.where(np.abs(dx) < 1.0, np.nan, dx)
    dv_dx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dx[:, None])
    du_dy = np.gradient(u, dy, axis=0)
    return dv_dx - du_dy


def nearest_lon_index(lons: np.ndarray, lon_value: float) -> int:
    diff = np.abs(lon_diff_deg(lons, lon_value))
    return int(np.nanargmin(diff))


def patch_values(field: np.ndarray, i: int, j: int, radius_cells: int = 2) -> np.ndarray:
    i0 = max(0, i - radius_cells)
    i1 = min(field.shape[0], i + radius_cells + 1)
    jj = [(j + k) % field.shape[1] for k in range(-radius_cells, radius_cells + 1)]
    return field[i0:i1, :][:, jj]


def split_for_year(year: int, splits: SplitYears = SplitYears()) -> str:
    if year <= splits.train_end:
        return "train"
    if year <= splits.val_end:
        return "val"
    return "test"


def extract_ibtracs_na_jas() -> pd.DataFrame:
    out_file = DATA_DIR / "ibtracs_na_jas_1991_2020_track_points.csv"
    ds = xr.open_dataset(IBTRACS, decode_times=False)
    sid = decode_bytes_array(ds["sid"].values)
    name = decode_bytes_array(ds["name"].values)
    season = ds["season"].values
    basin = decode_bytes_array(ds["basin"].values)
    nature = decode_bytes_array(ds["nature"].values)
    iso = decode_bytes_array(ds["iso_time"].values)
    lat = ds["lat"].values
    lon = lon360(ds["lon"].values)
    wmo_wind = ds["wmo_wind"].values if "wmo_wind" in ds else np.full_like(lat, np.nan)
    usa_wind = ds["usa_wind"].values if "usa_wind" in ds else np.full_like(lat, np.nan)
    wmo_pres = ds["wmo_pres"].values if "wmo_pres" in ds else np.full_like(lat, np.nan)
    usa_pres = ds["usa_pres"].values if "usa_pres" in ds else np.full_like(lat, np.nan)
    ds.close()

    rows = []
    for sidx in range(lat.shape[0]):
        yr = int(season[sidx]) if np.isfinite(season[sidx]) else -1
        if yr < 1991 or yr > 2020:
            continue
        sid_s = sid[sidx].strip()
        name_s = name[sidx].strip()
        for tidx in range(lat.shape[1]):
            iso_s = iso[sidx, tidx].strip()
            if not iso_s or iso_s.lower() == "nan":
                continue
            try:
                ts = pd.to_datetime(iso_s, utc=False)
            except Exception:
                continue
            if ts.year < 1991 or ts.year > 2020 or ts.month not in [7, 8, 9]:
                continue
            if ts.hour not in [0, 6, 12, 18]:
                continue
            basin_s = basin[sidx, tidx].strip()
            nature_s = nature[sidx, tidx].strip()
            if basin_s != "NA":
                continue
            la = float(lat[sidx, tidx])
            lo = float(lon[sidx, tidx])
            if not np.isfinite(la) or not np.isfinite(lo):
                continue
            if la < 0 or la > 50:
                continue
            wind = wmo_wind[sidx, tidx]
            if not np.isfinite(wind):
                wind = usa_wind[sidx, tidx]
            pres = wmo_pres[sidx, tidx]
            if not np.isfinite(pres):
                pres = usa_pres[sidx, tidx]
            if nature_s in {"ET", "SS"}:
                continue
            rows.append(
                {
                    "sample_id": f"IBT_{sid_s}_{ts.strftime('%Y%m%d%H')}",
                    "sid": sid_s,
                    "name": name_s,
                    "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": ts.strftime("%Y-%m-%d"),
                    "year": ts.year,
                    "month": ts.month,
                    "hour": ts.hour,
                    "basin": basin_s,
                    "nature": nature_s,
                    "lat": la,
                    "lon": lo,
                    "wind_kt": float(wind) if np.isfinite(wind) else np.nan,
                    "pres_hpa": float(pres) if np.isfinite(pres) else np.nan,
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(["sid", "time"]).sort_values(["sid", "time"])
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] IBTrACS NA JAS track points: {len(df)} -> {out_file}")
    return df


def make_observation_samples(track_df: pd.DataFrame, neg_ratio: float = 0.6, max_neg: int = 4500) -> pd.DataFrame:
    pos = track_df.copy()
    pos["source"] = "IBTRACS_ERA5"
    pos["experiment"] = "OBS"
    pos["label_tc"] = 1
    pos["candidate_lat"] = pos["lat"]
    pos["candidate_lon"] = pos["lon"]
    pos["matched_sid"] = pos["sid"]
    pos["matched_name"] = pos["name"]
    pos["label_distance_km"] = 0.0

    n_neg = min(int(len(pos) * neg_ratio), max_neg)
    era = xr.open_dataset(ERA5_ALL, decode_times=False)
    lats = era["lat"].values
    lons = era["lon"].values
    era.close()
    lat_choices = lats[(lats >= 0) & (lats <= 50)]
    lon_choices = np.concatenate([lons[(lons >= 260) & (lons <= 357.5)], lons[(lons >= 0) & (lons <= 30)]])
    pos_by_date = {
        d: g[["lat", "lon"]].to_numpy(dtype=float)
        for d, g in pos.groupby("date", sort=False)
    }
    dates = np.array(sorted(pos_by_date.keys()))
    neg_rows = []
    attempts = 0
    while len(neg_rows) < n_neg and attempts < n_neg * 80:
        attempts += 1
        d = str(RNG.choice(dates))
        la = float(RNG.choice(lat_choices))
        lo = float(RNG.choice(lon_choices))
        centers = pos_by_date.get(d)
        if centers is not None and len(centers):
            dist = haversine_km(la, lo, centers[:, 0], centers[:, 1])
            if np.nanmin(dist) < 1000.0:
                continue
            nearest = float(np.nanmin(dist))
        else:
            nearest = np.nan
        ts = pd.to_datetime(d)
        neg_rows.append(
            {
                "sample_id": f"NEG_{d.replace('-', '')}_{len(neg_rows):05d}",
                "sid": "",
                "name": "",
                "time": ts.strftime("%Y-%m-%d 00:00:00"),
                "date": d,
                "year": ts.year,
                "month": ts.month,
                "hour": 0,
                "basin": "NA",
                "nature": "BG",
                "lat": np.nan,
                "lon": np.nan,
                "wind_kt": np.nan,
                "pres_hpa": np.nan,
                "source": "IBTRACS_ERA5",
                "experiment": "OBS",
                "label_tc": 0,
                "candidate_lat": la,
                "candidate_lon": lo,
                "matched_sid": "",
                "matched_name": "",
                "label_distance_km": nearest,
            }
        )

    samples = pd.concat([pos, pd.DataFrame(neg_rows)], ignore_index=True)
    samples["split"] = samples["year"].astype(int).map(split_for_year)
    samples.to_csv(DATA_DIR / "obs_detection_samples_unfeatured.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] Observation samples before features: {len(samples)} positives={len(pos)} negatives={len(neg_rows)}")
    return samples


def era5_time_index(ds: xr.Dataset) -> dict[str, int]:
    hours = ds["time"].values.astype(float)
    dates = pd.Timestamp("1800-01-01") + pd.to_timedelta(hours, unit="h")
    return {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}


def extract_era5_features(samples: pd.DataFrame) -> pd.DataFrame:
    out_file = DATA_DIR / "obs_detection_samples_features.csv"
    ds = xr.open_dataset(ERA5_ALL, decode_times=False)
    global_time_map = era5_time_index(ds)
    idx_to_date = {idx: date for date, idx in global_time_map.items()}
    all_lats = ds["lat"].values
    all_lons = ds["lon"].values
    levels = ds["level"].values.astype(int)
    level_indices = [int(np.where(levels == level)[0][0]) for level in LEVELS]
    lat_indices = np.where((all_lats >= -5) & (all_lats <= 55))[0]
    part_files = []

    samples_by_year = list(samples.groupby("year", sort=True))
    for n_year, (year, year_group) in enumerate(samples_by_year, start=1):
        year = int(year)
        part_file = DATA_DIR / f"obs_detection_samples_features_year{year}.csv"
        if part_file.exists():
            part_files.append(part_file)
            print(f"[ERA5] reuse cached year={year} -> {part_file}", flush=True)
            continue
        dates_for_year = sorted(set(str(x) for x in year_group["date"]))
        time_indices = [global_time_map[d] for d in dates_for_year if d in global_time_map]
        if not time_indices:
            continue
        # Load a compact JAS/NA latitude strip for the whole year. This avoids
        # thousands of tiny reads from the 22 GB ERA5 file.
        sub = ds[["u", "v", "T", "q"]].isel(
            time=time_indices,
            level=level_indices,
            lat=lat_indices,
        ).load()
        lats = sub["lat"].values
        lons = sub["lon"].values
        sub_dates = [idx_to_date[idx] for idx in time_indices]
        sub_date_to_pos = {d: i for i, d in enumerate(sub_dates)}
        level_idx = {level: pos for pos, level in enumerate(LEVELS)}

        year_records = []
        grouped = list(year_group.groupby("date", sort=True))
        for date, group in grouped:
            if date not in sub_date_to_pos:
                continue
            slab = sub.isel(time=sub_date_to_pos[date])
            u_all = slab["u"].values
            v_all = slab["v"].values
            t_all = slab["T"].values
            q_all = slab["q"].values

            fields: dict[str, np.ndarray] = {}
            for lev in LEVELS:
                li = level_idx[lev]
                fields[f"u_{lev}"] = u_all[li]
                fields[f"v_{lev}"] = v_all[li]
                fields[f"t_{lev}"] = t_all[li]
                fields[f"q_{lev}"] = q_all[li]
                fields[f"rh_{lev}"] = rh_from_q_t_p(q_all[li], t_all[li], lev)
            zeta850 = compute_zeta(fields["u_850"], fields["v_850"], lats, lons)
            zeta500 = compute_zeta(fields["u_500"], fields["v_500"], lats, lons)
            shear = np.hypot(fields["u_200"] - fields["u_850"], fields["v_200"] - fields["v_850"])
            steer_u = np.nanmean(np.stack([fields[f"u_{lev}"] for lev in LEVELS]), axis=0)
            steer_v = np.nanmean(np.stack([fields[f"v_{lev}"] for lev in LEVELS]), axis=0)

            for row in group.to_dict("records"):
                la = float(row["candidate_lat"])
                lo = float(row["candidate_lon"])
                i = int(np.nanargmin(np.abs(lats - la)))
                j = nearest_lon_index(lons, lo)
                rec = dict(row)
                rec["lat"] = la
                rec.update(
                    {
                        "lon_sin": math.sin(math.radians(lo)),
                        "lon_cos": math.cos(math.radians(lo)),
                        "month_sin": math.sin(2 * math.pi * int(row["month"]) / 12.0),
                        "month_cos": math.cos(2 * math.pi * int(row["month"]) / 12.0),
                    }
                )
                for lev in LEVELS:
                    rec[f"u_{lev}"] = float(fields[f"u_{lev}"][i, j])
                    rec[f"v_{lev}"] = float(fields[f"v_{lev}"][i, j])
                    rec[f"t_{lev}"] = float(fields[f"t_{lev}"][i, j])
                    rec[f"q_{lev}"] = float(fields[f"q_{lev}"][i, j])
                    rec[f"rh_{lev}"] = float(fields[f"rh_{lev}"][i, j])
                rec["zeta_850"] = float(zeta850[i, j])
                rec["zeta_500"] = float(zeta500[i, j])
                rec["wshear_850_200"] = float(shear[i, j])
                rec["steer_u_850_200"] = float(steer_u[i, j])
                rec["steer_v_850_200"] = float(steer_v[i, j])
                rec["zeta850_max_r500"] = float(np.nanmax(patch_values(zeta850, i, j, 2)))
                rec["zeta500_max_r500"] = float(np.nanmax(patch_values(zeta500, i, j, 2)))
                rec["rh700_mean_r500"] = float(np.nanmean(patch_values(fields["rh_700"], i, j, 2)))
                rec["shear_min_r500"] = float(np.nanmin(patch_values(shear, i, j, 2)))
                year_records.append(rec)

        part_df = pd.DataFrame(year_records)
        part_df.to_csv(part_file, index=False, encoding="utf-8-sig")
        part_files.append(part_file)
        print(f"[ERA5] feature years {n_year}/{len(samples_by_year)} year={year} records={len(part_df)}", flush=True)

    ds.close()
    df = pd.concat([pd.read_csv(p) for p in part_files], ignore_index=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label_tc"])
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] ERA5 feature samples: {len(df)} -> {out_file}")
    return df


def jas_time_indices() -> np.ndarray:
    return np.arange(181 * 4, 273 * 4)


def na_domain_indices(lats: np.ndarray, lons: np.ndarray):
    lat_idx = np.where((lats >= 0) & (lats <= 50))[0]
    lon_idx = np.where((lons >= 260) | (lons <= 30))[0]
    return lat_idx, lon_idx


def extract_cesm_background_samples(n_samples: int = 700) -> pd.DataFrame:
    out_file = DATA_DIR / "cesm_ctl_background_samples.csv"
    years = list(range(31, 46))
    per_year_target = max(1, math.ceil(n_samples / len(years)))
    records = []
    for year in years:
        if len(records) >= n_samples:
            break
        path = CESM_CTL_PRE / f"cesm_{year}.nc"
        if not path.exists():
            continue
        ds = xr.open_dataset(path, decode_times=False)
        lats = ds["lat"].values
        lons = ds["lon"].values
        plev = ds["plev"].values.astype(int)
        pidx = {200: int(np.where(plev == 20000)[0][0]), 300: int(np.where(plev == 30000)[0][0]),
                500: int(np.where(plev == 50000)[0][0]), 700: int(np.where(plev == 70000)[0][0]),
                850: int(np.where(plev == 85000)[0][0])}
        lat_idx, lon_idx = na_domain_indices(lats, lons)
        year_added = 0
        time_choices = RNG.choice(jas_time_indices(), size=12, replace=False)
        for t in time_choices:
            if len(records) >= n_samples:
                break
            u = ds["U"].isel(time=t).values
            v = ds["V"].isel(time=t).values
            temp = ds["T"].isel(time=t).values
            q = ds["Q"].isel(time=t).values
            rh = ds["RH"].isel(time=t).values
            zeta = ds["zeta"].isel(time=t).values
            shear = ds["Wshear_850_200"].isel(time=t).values
            owz850 = ds["OWZ"].isel(time=t, plev=pidx[850]).values
            candidate_pairs = []
            for _ in range(120):
                i = int(RNG.choice(lat_idx))
                j = int(RNG.choice(lon_idx))
                if np.isfinite(owz850[i, j]) and owz850[i, j] < 25e-6:
                    candidate_pairs.append((i, j))
                if len(candidate_pairs) >= 6:
                    break
            for i, j in candidate_pairs:
                if len(records) >= n_samples or year_added >= per_year_target:
                    break
                month = 7 + int((t - 181 * 4) // (31 * 4))
                la = float(lats[i])
                lo = float(lons[j])
                rec = {
                    "sample_id": f"CESM_CTL_BG_{year}_{t}_{i}_{j}",
                    "source": "CESM_CTL_BG",
                    "experiment": "CTL",
                    "model_year": year,
                    "year": 2000,
                    "month": min(month, 9),
                    "hour": int((t % 4) * 6),
                    "lat": la,
                    "candidate_lat": la,
                    "candidate_lon": lo,
                    "label_tc": 0,
                    "split": "train",
                    "lon_sin": math.sin(math.radians(lo)),
                    "lon_cos": math.cos(math.radians(lo)),
                    "month_sin": math.sin(2 * math.pi * min(month, 9) / 12.0),
                    "month_cos": math.cos(2 * math.pi * min(month, 9) / 12.0),
                }
                for lev in LEVELS:
                    pi = pidx[lev]
                    rec[f"u_{lev}"] = float(u[pi, i, j])
                    rec[f"v_{lev}"] = float(v[pi, i, j])
                    rec[f"t_{lev}"] = float(temp[pi, i, j])
                    rec[f"q_{lev}"] = float(q[pi, i, j])
                    rec[f"rh_{lev}"] = float(rh[pi, i, j])
                rec["zeta_850"] = float(zeta[pidx[850], i, j])
                rec["zeta_500"] = float(zeta[pidx[500], i, j])
                rec["wshear_850_200"] = float(shear[i, j])
                rec["steer_u_850_200"] = float(np.nanmean([u[pidx[lev], i, j] for lev in LEVELS]))
                rec["steer_v_850_200"] = float(np.nanmean([v[pidx[lev], i, j] for lev in LEVELS]))
                rec["zeta850_max_r500"] = float(np.nanmax(patch_values(zeta[pidx[850]], i, j, 2)))
                rec["zeta500_max_r500"] = float(np.nanmax(patch_values(zeta[pidx[500]], i, j, 2)))
                rec["rh700_mean_r500"] = float(np.nanmean(patch_values(rh[pidx[700]], i, j, 2)))
                rec["shear_min_r500"] = float(np.nanmin(patch_values(shear, i, j, 2)))
                records.append(rec)
                year_added += 1
            if year_added >= per_year_target:
                break
        ds.close()
        print(f"[CESM BG] year {year}: added {year_added}, total {len(records)}", flush=True)

    df = pd.DataFrame(records).dropna(subset=FEATURE_COLUMNS + ["label_tc"])
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] CESM CTL background negatives: {len(df)} -> {out_file}")
    return df


def build_training_table(obs_features: pd.DataFrame, cesm_bg: pd.DataFrame) -> pd.DataFrame:
    obs = obs_features.copy()
    obs["model_year"] = ""
    keep = sorted(set(["sample_id", "source", "experiment", "model_year", "year", "month", "hour",
                       "candidate_lat", "candidate_lon", "label_tc", "split", "sid", "name", "time",
                       "wind_kt", "pres_hpa"]) | set(FEATURE_COLUMNS))
    for col in keep:
        if col not in obs:
            obs[col] = ""
        if col not in cesm_bg:
            cesm_bg[col] = ""
    train = pd.concat([obs[keep], cesm_bg[keep]], ignore_index=True)
    train = train.dropna(subset=FEATURE_COLUMNS + ["label_tc"])
    train["label_tc"] = train["label_tc"].astype(int)
    out_file = DATA_DIR / "training_detection_samples.csv"
    train.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] Training table: {len(train)} -> {out_file}")
    return train


def fit_detection_models(train: pd.DataFrame) -> pd.DataFrame:
    model_specs = {
        "logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "svc_linear": Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="linear", probability=False, class_weight="balanced", C=1.0))]),
        "svc_rbf": Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", probability=False, class_weight="balanced", C=2.0, gamma="scale"))]),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=404, class_weight="balanced_subsample", min_samples_leaf=3, n_jobs=-1),
    }
    train_df = train[train["split"] == "train"].copy()
    test_df = train[(train["split"] == "test") & (train["experiment"] == "OBS")].copy()

    if len(train_df) > 9000:
        parts = []
        for label, group in train_df.groupby("label_tc"):
            n_label = max(1, int(round(9000 * len(group) / len(train_df))))
            parts.append(group.sample(n=min(len(group), n_label), random_state=404))
        train_df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=404)

    X_train = train_df[FEATURE_COLUMNS].astype(float)
    y_train = train_df["label_tc"].astype(int)
    X_test = test_df[FEATURE_COLUMNS].astype(float)
    y_test = test_df["label_tc"].astype(int)

    rows = []
    best_name = None
    best_f1 = -np.inf
    for name, model in model_specs.items():
        print(f"[TRAIN] detection model {name}")
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            dec = model.decision_function(X_test)
            prob = 1 / (1 + np.exp(-dec))
        pred = (prob >= 0.5).astype(int)
        row = {
            "model": name,
            "n_train": len(X_train),
            "n_test_obs": len(X_test),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "average_precision": average_precision_score(y_test, prob),
            "brier": brier_score_loss(y_test, prob),
        }
        rows.append(row)
        joblib.dump(model, MODEL_DIR / f"detection_{name}.joblib")
        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            best_name = name

    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics.to_csv(REPORT_DIR / "detection_model_metrics.csv", index=False, encoding="utf-8-sig")
    if best_name:
        shutil.copy2(MODEL_DIR / f"detection_{best_name}.joblib", MODEL_DIR / "detection_best.joblib")
    plot_detection_metrics(metrics)
    print(f"[OK] Detection metrics -> {REPORT_DIR / 'detection_model_metrics.csv'}")
    return metrics


def plot_detection_metrics(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    x = np.arange(len(metrics))
    ax.bar(x - 0.2, metrics["f1"], width=0.4, label="F1", color="#267c8f")
    ax.bar(x + 0.2, metrics["roc_auc"], width=0.4, label="ROC-AUC", color="#c7634d")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("TC detection skill on held-out IBTrACS/ERA5 test years")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "detection_model_metrics.png")
    plt.close(fig)


def build_path_samples(obs_features: pd.DataFrame) -> pd.DataFrame:
    pos = obs_features[obs_features["label_tc"].astype(int) == 1].copy()
    pos["dt"] = pd.to_datetime(pos["time"])
    pos = pos.sort_values(["sid", "dt"])
    by_key = {(r.sid, r.dt): r for r in pos.itertuples(index=False)}
    rows = []
    for r in pos.itertuples(index=False):
        prev6 = by_key.get((r.sid, r.dt - pd.Timedelta(hours=6)))
        prev12 = by_key.get((r.sid, r.dt - pd.Timedelta(hours=12)))
        if prev6 is None or prev12 is None:
            continue
        dx6, dy6 = dxdy_km(prev6.candidate_lat, prev6.candidate_lon, r.candidate_lat, r.candidate_lon)
        dx12, dy12 = dxdy_km(prev12.candidate_lat, prev12.candidate_lon, r.candidate_lat, r.candidate_lon)
        for lead in [6, 12, 18, 24]:
            fut = by_key.get((r.sid, r.dt + pd.Timedelta(hours=lead)))
            if fut is None:
                continue
            dx, dy = dxdy_km(r.candidate_lat, r.candidate_lon, fut.candidate_lat, fut.candidate_lon)
            row = {
                "forecast_id": f"{r.sid}_{r.dt.strftime('%Y%m%d%H')}_{lead}",
                "sid": r.sid,
                "init_time": r.dt.strftime("%Y-%m-%d %H:%M:%S"),
                "year": int(r.year),
                "month": int(r.month),
                "lead_h": lead,
                "init_lat": float(r.candidate_lat),
                "init_lon": float(r.candidate_lon),
                "target_lat": float(fut.candidate_lat),
                "target_lon": float(fut.candidate_lon),
                "target_dx_km": float(dx),
                "target_dy_km": float(dy),
                "motion_dx_6h": float(dx6),
                "motion_dy_6h": float(dy6),
                "motion_dx_12h": float(dx12),
                "motion_dy_12h": float(dy12),
                "wind_kt": float(r.wind_kt) if pd.notna(r.wind_kt) else np.nan,
                "pres_hpa": float(r.pres_hpa) if pd.notna(r.pres_hpa) else np.nan,
                "split": split_for_year(int(r.year)),
            }
            for col in FEATURE_COLUMNS:
                row[col] = getattr(r, col)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["wind_kt"] = df["wind_kt"].fillna(df["wind_kt"].median())
    df["pres_hpa"] = df["pres_hpa"].fillna(df["pres_hpa"].median())
    df = df.dropna()
    out_file = DATA_DIR / "svr_path_samples.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] SVR path samples: {len(df)} -> {out_file}")
    return df


def fit_path_models(path_samples: pd.DataFrame) -> pd.DataFrame:
    path_features = [
        "init_lat",
        "lon_sin",
        "lon_cos",
        "month_sin",
        "month_cos",
        "motion_dx_6h",
        "motion_dy_6h",
        "motion_dx_12h",
        "motion_dy_12h",
        "wind_kt",
        "pres_hpa",
        "steer_u_850_200",
        "steer_v_850_200",
        "wshear_850_200",
        "zeta_850",
        "zeta_500",
        "rh_700",
        "rh700_mean_r500",
    ]
    kernels = ["linear", "rbf", "poly", "sigmoid"]
    rows = []
    for lead in [6, 12, 18, 24]:
        sub = path_samples[path_samples["lead_h"] == lead].copy()
        train = sub[sub["split"] == "train"].copy()
        test = sub[sub["split"] == "test"].copy()
        if len(train) > 2500:
            train = train.sample(n=2500, random_state=lead)
        X_train = train[path_features].astype(float)
        ydx_train = train["target_dx_km"].astype(float)
        ydy_train = train["target_dy_km"].astype(float)
        X_test = test[path_features].astype(float)
        ydx_test = test["target_dx_km"].astype(float).to_numpy()
        ydy_test = test["target_dy_km"].astype(float).to_numpy()

        # Baselines
        pred_dx = test["motion_dx_6h"].to_numpy() * (lead / 6.0)
        pred_dy = test["motion_dy_6h"].to_numpy() * (lead / 6.0)
        ate = np.hypot(pred_dx - ydx_test, pred_dy - ydy_test)
        rows.append(
            {
                "lead_h": lead,
                "model": "persistence_motion",
                "kernel": "baseline",
                "n_train": len(train),
                "n_test": len(test),
                "ate_mean_km": float(np.nanmean(ate)),
                "ate_median_km": float(np.nanmedian(ate)),
            }
        )
        lr_x = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
        lr_y = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
        lr_x.fit(X_train, ydx_train)
        lr_y.fit(X_train, ydy_train)
        pdx = lr_x.predict(X_test)
        pdy = lr_y.predict(X_test)
        ate = np.hypot(pdx - ydx_test, pdy - ydy_test)
        rows.append(
            {
                "lead_h": lead,
                "model": "linear_regression",
                "kernel": "linear",
                "n_train": len(train),
                "n_test": len(test),
                "ate_mean_km": float(np.nanmean(ate)),
                "ate_median_km": float(np.nanmedian(ate)),
            }
        )
        joblib.dump({"x": lr_x, "y": lr_y, "features": path_features}, MODEL_DIR / f"path_lr_lead{lead}.joblib")

        for kernel in kernels:
            print(f"[TRAIN] SVR path lead={lead} kernel={kernel}")
            svr_x = Pipeline([("scale", StandardScaler()), ("model", SVR(kernel=kernel, C=10.0, epsilon=20.0, gamma="scale"))])
            svr_y = Pipeline([("scale", StandardScaler()), ("model", SVR(kernel=kernel, C=10.0, epsilon=20.0, gamma="scale"))])
            svr_x.fit(X_train, ydx_train)
            svr_y.fit(X_train, ydy_train)
            pdx = svr_x.predict(X_test)
            pdy = svr_y.predict(X_test)
            ate = np.hypot(pdx - ydx_test, pdy - ydy_test)
            rows.append(
                {
                    "lead_h": lead,
                    "model": "SVR",
                    "kernel": kernel,
                    "n_train": len(train),
                    "n_test": len(test),
                    "ate_mean_km": float(np.nanmean(ate)),
                    "ate_median_km": float(np.nanmedian(ate)),
                }
            )
            joblib.dump(
                {"x": svr_x, "y": svr_y, "features": path_features},
                MODEL_DIR / f"path_svr_{kernel}_lead{lead}.joblib",
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORT_DIR / "svr_path_metrics.csv", index=False, encoding="utf-8-sig")
    plot_path_metrics(metrics)
    print(f"[OK] SVR path metrics -> {REPORT_DIR / 'svr_path_metrics.csv'}")
    return metrics


def plot_path_metrics(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    for name, group in metrics.groupby(["model", "kernel"]):
        label = name[0] if name[0] != "SVR" else f"SVR-{name[1]}"
        g = group.sort_values("lead_h")
        ax.plot(g["lead_h"], g["ate_mean_km"], marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("forecast lead (h)")
    ax.set_ylabel("mean ATE (km)")
    ax.set_title("Short-term TC track forecast error")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "svr_path_ate_by_lead.png")
    plt.close(fig)


def write_year100_exclusion_manifest() -> None:
    rows = []
    for exp, years, pre_dir, owz_dir in [
        ("CTL", range(31, 61), CESM_CTL_PRE, OWZ_CTL),
        ("GGW", [y for y in range(90, 120) if y != 100], CESM_GGW_PRE, OWZ_GGW),
    ]:
        for y in years:
            rows.append(
                {
                    "experiment": exp,
                    "year": y,
                    "preprocessed_file": str(pre_dir / f"cesm_{y}.nc"),
                    "owz_track_file": str(owz_dir / f"systems_all_year{y}.nc"),
                    "included": True,
                    "reason": "GGW year100 excluded globally" if exp == "GGW" else "included",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "included_cesm_years_excluding_ggw100.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    copy_this_script()
    write_year100_exclusion_manifest()
    track = extract_ibtracs_na_jas()
    obs_samples = make_observation_samples(track)
    obs_features = extract_era5_features(obs_samples)
    cesm_bg = extract_cesm_background_samples()
    training = build_training_table(obs_features, cesm_bg)
    det_metrics = fit_detection_models(training)
    path_samples = build_path_samples(obs_features)
    path_metrics = fit_path_models(path_samples)

    summary = {
        "year100_policy": "GGW year100 excluded from all CESM manifests and OWZ comparisons.",
        "files": {
            "ibtracs_track_points": str(DATA_DIR / "ibtracs_na_jas_1991_2020_track_points.csv"),
            "obs_detection_features": str(DATA_DIR / "obs_detection_samples_features.csv"),
            "cesm_ctl_background_samples": str(DATA_DIR / "cesm_ctl_background_samples.csv"),
            "training_detection_samples": str(DATA_DIR / "training_detection_samples.csv"),
            "svr_path_samples": str(DATA_DIR / "svr_path_samples.csv"),
            "detection_metrics": str(REPORT_DIR / "detection_model_metrics.csv"),
            "path_metrics": str(REPORT_DIR / "svr_path_metrics.csv"),
        },
        "n": {
            "ibtracs_track_points": int(len(track)),
            "obs_feature_samples": int(len(obs_features)),
            "cesm_ctl_background_samples": int(len(cesm_bg)),
            "training_detection_samples": int(len(training)),
            "path_samples": int(len(path_samples)),
        },
        "best_detection": det_metrics.iloc[0].to_dict() if len(det_metrics) else {},
        "best_svr_by_lead": path_metrics[path_metrics["model"] == "SVR"]
        .sort_values(["lead_h", "ate_mean_km"])
        .groupby("lead_h")
        .head(1)
        .to_dict(orient="records"),
    }
    (REPORT_DIR / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] summary -> {REPORT_DIR / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()

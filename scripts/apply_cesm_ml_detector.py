from __future__ import annotations

import json
import math
import os
import shutil
import warnings
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import maximum_filter


warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_ROOT = Path(os.environ.get("TC_DETECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_DIR = OUT_ROOT / "data"
MODEL_DIR = OUT_ROOT / "models"
FIG_DIR = OUT_ROOT / "figures"
REPORT_DIR = OUT_ROOT / "reports"
SCRIPT_DIR = OUT_ROOT / "scripts"
APPLY_DIR = DATA_DIR / "cesm_ml_application"
YEAR_DIR = APPLY_DIR / "year_tracks"

CESM_CTL_PRE = Path(os.environ.get("CESM_CTL_PRE", r"E:\path\to\CESM\control"))
CESM_GGW_PRE = Path(os.environ.get("CESM_GGW_PRE", r"G:\path\to\CESM\surface"))
OWZ_CTL = Path(os.environ.get("OWZ_CTL", r"E:\path\to\OWZ\control"))
OWZ_GGW = Path(os.environ.get("OWZ_GGW", r"G:\path\to\OWZ\surface"))

EARTH_R_KM = 6371.0
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


def ensure_dirs() -> None:
    for path in [APPLY_DIR, YEAR_DIR, FIG_DIR, REPORT_DIR, SCRIPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def copy_this_script() -> None:
    src = Path(__file__).resolve()
    dst = SCRIPT_DIR / src.name
    if src != dst:
        shutil.copy2(src, dst)


def lon_diff_deg(lon2, lon1):
    return (np.asarray(lon2) - np.asarray(lon1) + 180.0) % 360.0 - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon_diff_deg(lon2, lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return EARTH_R_KM * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def month_from_time_index(t: int) -> int:
    # CESM no-leap year, four 6-hourly samples per day.
    day = t // 4
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    acc = 0
    for idx, nday in enumerate(month_lengths, start=1):
        if day < acc + nday:
            return idx
        acc += nday
    return 12


def jas_indices() -> np.ndarray:
    return np.arange(181 * 4, 273 * 4)


def na_domain_lons(lons: np.ndarray) -> np.ndarray:
    return np.concatenate([np.where(lons >= 260.0)[0], np.where(lons <= 30.0)[0]])


def plev_index(plev: np.ndarray, hpa: int) -> int:
    target = hpa * 100.0
    return int(np.nanargmin(np.abs(plev - target)))


def patch(field: np.ndarray, i: int, j: int, r: int = 2) -> np.ndarray:
    i0 = max(0, i - r)
    i1 = min(field.shape[0], i + r + 1)
    j0 = max(0, j - r)
    j1 = min(field.shape[1], j + r + 1)
    return field[i0:i1, j0:j1]


def model_probability(model, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)[:, 1]
        score = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / np.clip(1 - prob, 1e-6, 1 - 1e-6))
        return prob, score
    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
    else:
        pred = model.predict(x)
        score = np.asarray(pred, dtype=float)
    prob = 1.0 / (1.0 + np.exp(-score))
    return prob, score


def nms_by_distance(df: pd.DataFrame, dist_km: float = 800.0) -> pd.DataFrame:
    if df.empty:
        return df
    kept = []
    used = np.zeros(len(df), dtype=bool)
    order = np.argsort(-df["prob_tc"].to_numpy())
    vals = df.reset_index(drop=True)
    for idx in order:
        if used[idx]:
            continue
        row = vals.iloc[idx]
        kept.append(row)
        d = haversine_km(row["lat"], row["lon"], vals["lat"].to_numpy(), vals["lon"].to_numpy())
        used |= d <= dist_km
    return pd.DataFrame(kept)


def track_candidates(cand: pd.DataFrame, min_steps: int = 8, max_link_km: float = 800.0) -> pd.DataFrame:
    if cand.empty:
        return cand
    cand = cand.sort_values(["time_index", "prob_tc"], ascending=[True, False]).reset_index(drop=True)
    tracks: list[list[dict]] = []
    active: dict[int, dict] = {}
    next_id = 1
    for t, cur in cand.groupby("time_index", sort=True):
        cur = nms_by_distance(cur, 800.0).reset_index(drop=True)
        cur_records = [r for r in cur.to_dict("records")]
        if not active:
            for r in cur_records:
                r["track_id"] = next_id
                tracks.append([r])
                active[next_id] = r
                next_id += 1
            continue
        matched_cur: set[int] = set()
        new_active: dict[int, dict] = {}
        for tid, last in sorted(active.items()):
            if not cur_records:
                continue
            d = np.array(
                [haversine_km(last["lat"], last["lon"], r["lat"], r["lon"]) for r in cur_records],
                dtype=float,
            )
            order = np.argsort(d)
            for ci in order:
                if ci in matched_cur:
                    continue
                if d[ci] <= max_link_km:
                    r = cur_records[ci]
                    r["track_id"] = tid
                    tracks[tid - 1].append(r)
                    new_active[tid] = r
                    matched_cur.add(ci)
                    break
        for ci, r in enumerate(cur_records):
            if ci not in matched_cur:
                r["track_id"] = next_id
                tracks.append([r])
                new_active[next_id] = r
                next_id += 1
        active = new_active
    rows = []
    new_id = 1
    for tr in tracks:
        if len(tr) < min_steps:
            continue
        for life_step, r in enumerate(tr, start=1):
            r = dict(r)
            r["track_id"] = new_id
            r["life_step"] = life_step
            r["lifetime_steps"] = len(tr)
            rows.append(r)
        new_id += 1
    return pd.DataFrame(rows)


def build_features_for_candidates(
    year: int,
    t: int,
    ds_t: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
    pidx: dict[int, int],
    candidate_pairs: Iterable[tuple[int, int]],
) -> pd.DataFrame:
    u = ds_t["U"].values
    v = ds_t["V"].values
    temp = ds_t["T"].values
    q = ds_t["Q"].values
    rh = ds_t["RH"].values
    zeta = ds_t["zeta"].values
    shear = ds_t["Wshear_850_200"].values
    month = month_from_time_index(t)
    zeta850 = zeta[pidx[850]]
    zeta500 = zeta[pidx[500]]
    rh700 = rh[pidx[700]]
    rows = []
    for i, j in candidate_pairs:
        la = float(lats[i])
        lo = float(lons[j])
        rec = {
            "year": year,
            "time_index": int(t),
            "month": month,
            "hour": int((t % 4) * 6),
            "lat": la,
            "lon": lo,
            "lon_sin": math.sin(math.radians(lo)),
            "lon_cos": math.cos(math.radians(lo)),
            "month_sin": math.sin(2.0 * math.pi * month / 12.0),
            "month_cos": math.cos(2.0 * math.pi * month / 12.0),
        }
        for lev in LEVELS:
            pi = pidx[lev]
            rec[f"u_{lev}"] = float(u[pi, i, j])
            rec[f"v_{lev}"] = float(v[pi, i, j])
            rec[f"t_{lev}"] = float(temp[pi, i, j])
            rec[f"q_{lev}"] = float(q[pi, i, j])
            rec[f"rh_{lev}"] = float(rh[pi, i, j])
        rec["zeta_850"] = float(zeta850[i, j])
        rec["zeta_500"] = float(zeta500[i, j])
        rec["wshear_850_200"] = float(shear[i, j])
        rec["steer_u_850_200"] = float(np.nanmean([u[pidx[lev], i, j] for lev in LEVELS]))
        rec["steer_v_850_200"] = float(np.nanmean([v[pidx[lev], i, j] for lev in LEVELS]))
        rec["zeta850_max_r500"] = float(np.nanmax(patch(zeta850, i, j, 2)))
        rec["zeta500_max_r500"] = float(np.nanmax(patch(zeta500, i, j, 2)))
        rec["rh700_mean_r500"] = float(np.nanmean(patch(rh700, i, j, 2)))
        rec["shear_min_r500"] = float(np.nanmin(patch(shear, i, j, 2)))
        rows.append(rec)
    return pd.DataFrame(rows)


def year_ml_tracks(experiment: str, year: int, model, threshold: float = 0.55) -> pd.DataFrame:
    out_file = YEAR_DIR / f"ml_tracks_{experiment}_year{year}.csv"
    if out_file.exists():
        print(f"[ML] reuse {experiment} {year}: {out_file}", flush=True)
        return pd.read_csv(out_file)
    pre_dir = CESM_CTL_PRE if experiment == "CTL" else CESM_GGW_PRE
    path = pre_dir / f"cesm_{year}.nc"
    if not path.exists():
        print(f"[ML] missing {path}", flush=True)
        return pd.DataFrame()

    ds = xr.open_dataset(path, decode_times=False, engine="h5netcdf")
    lats_full = ds["lat"].values
    lons_full = ds["lon"].values
    plev_full = ds["plev"].values
    lat_idx = np.where((lats_full >= 0.0) & (lats_full <= 50.0))[0]
    lon_idx = na_domain_lons(lons_full)
    pidx_full = {lev: plev_index(plev_full, lev) for lev in LEVELS}
    time_values = jas_indices()
    # Load one compact seasonal North Atlantic cube per model year. Reading
    # time-slice-by-time-slice from these large NetCDF files is prohibitively slow.
    ds_dom = ds[
        ["U", "V", "T", "Q", "RH", "zeta", "Wshear_850_200"]
    ].isel(
        time=time_values,
        plev=[pidx_full[lev] for lev in LEVELS],
        lat=lat_idx,
        lon=lon_idx,
    ).load()
    ds.close()
    lats = ds_dom["lat"].values
    lons = ds_dom["lon"].values
    local_pidx = {lev: pos for pos, lev in enumerate(LEVELS)}

    all_candidates = []
    for pos, t in enumerate(time_values):
        ds_t = ds_dom.isel(time=pos)
        zeta850 = ds_t["zeta"].isel(plev=local_pidx[850]).values
        zeta500 = ds_t["zeta"].isel(plev=local_pidx[500]).values
        shear = ds_t["Wshear_850_200"].values
        rh700 = ds_t["RH"].isel(plev=local_pidx[700]).values
        local_max = zeta850 >= maximum_filter(zeta850, size=3, mode="nearest")
        mask = (
            np.isfinite(zeta850)
            & local_max
            & (zeta850 > 1.0e-5)
            & (zeta500 > 0.0)
            & (shear < 30.0)
            & (rh700 > 35.0)
        )
        pairs = np.argwhere(mask)
        if pairs.size == 0:
            continue
        strength = zeta850[pairs[:, 0], pairs[:, 1]]
        order = np.argsort(-strength)[:24]
        pairs_t = [(int(pairs[k, 0]), int(pairs[k, 1])) for k in order]
        feat = build_features_for_candidates(year, int(t), ds_t, lats, lons, local_pidx, pairs_t)
        if feat.empty:
            continue
        prob, score = model_probability(model, feat[FEATURE_COLUMNS].astype(float))
        feat["prob_tc"] = prob
        feat["decision_score"] = score
        feat = feat[feat["prob_tc"] >= threshold].copy()
        if not feat.empty:
            all_candidates.append(feat)

    cand = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    tracks = track_candidates(cand, min_steps=8, max_link_km=800.0)
    if not tracks.empty:
        tracks.insert(0, "experiment", experiment)
        tracks["global_track_id"] = tracks.apply(
            lambda r: f"ML_{experiment}_{year}_{int(r['track_id']):04d}", axis=1
        )
    tracks.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(
        f"[ML] {experiment} {year}: candidates={len(cand)} track_points={len(tracks)} tracks={tracks['global_track_id'].nunique() if not tracks.empty else 0}",
        flush=True,
    )
    return tracks


def apply_ml_all_years() -> pd.DataFrame:
    model = joblib.load(MODEL_DIR / "detection_best.joblib")
    rows = []
    for experiment, years in [
        ("CTL", list(range(31, 61))),
        ("GGW", [y for y in range(90, 120) if y != 100]),
    ]:
        for year in years:
            rows.append(year_ml_tracks(experiment, year, model))
    all_tracks = pd.concat([r for r in rows if r is not None and not r.empty], ignore_index=True)
    out = APPLY_DIR / "ml_detected_track_points.csv"
    all_tracks.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] ML all track points: {len(all_tracks)} -> {out}", flush=True)
    return all_tracks


def read_owz_year(experiment: str, year: int) -> pd.DataFrame:
    trk_dir = OWZ_CTL if experiment == "CTL" else OWZ_GGW
    pre_dir = CESM_CTL_PRE if experiment == "CTL" else CESM_GGW_PRE
    trk_file = trk_dir / f"systems_all_year{year}.nc"
    pre_file = pre_dir / f"cesm_{year}.nc"
    if not trk_file.exists() or not pre_file.exists():
        return pd.DataFrame()
    ds_pre = xr.open_dataset(pre_file, decode_times=False)
    time_to_idx = {int(v): i for i, v in enumerate(ds_pre["time"].values)}
    ds_pre.close()
    ds = xr.open_dataset(trk_file, decode_times=False)
    df = pd.DataFrame(
        {
            "experiment": experiment,
            "year": year,
            "global_track_id": [f"OWZ_{experiment}_{year}_{int(x):04d}" for x in ds["system_id"].values],
            "track_id": ds["system_id"].values.astype(int),
            "time": ds["time"].values.astype(int),
            "lat": ds["lat"].values.astype(float),
            "lon": np.mod(ds["lon"].values.astype(float), 360.0),
            "windMax": ds["windMax"].values.astype(float),
            "twarm": ds["twarm"].values.astype(float),
            "jetcore": ds["jetcore"].values.astype(int),
        }
    )
    ds.close()
    df["time_index"] = df["time"].map(time_to_idx)
    df = df.dropna(subset=["time_index"]).copy()
    df["time_index"] = df["time_index"].astype(int)
    df["month"] = df["time_index"].map(month_from_time_index)
    df = df[df["month"].isin([7, 8, 9])].copy()
    df = df[(df["lat"] >= 0) & (df["lat"] <= 50) & ((df["lon"] >= 260) | (df["lon"] <= 30))].copy()
    return df


def read_owz_all() -> pd.DataFrame:
    rows = []
    for experiment, years in [
        ("CTL", list(range(31, 61))),
        ("GGW", [y for y in range(90, 120) if y != 100]),
    ]:
        for year in years:
            rows.append(read_owz_year(experiment, year))
    df = pd.concat([r for r in rows if r is not None and not r.empty], ignore_index=True)
    out = APPLY_DIR / "owz_track_points_jas_na_excluding_ggw100.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] OWZ JAS NA track points: {len(df)} -> {out}", flush=True)
    return df


def point_in_mdr(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return (lat >= 10) & (lat <= 20) & (lon >= 290) & (lon <= 342.5)


def point_in_box1(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return (lat >= 10) & (lat <= 30) & (lon >= 270) & (lon <= 300)


def point_in_box2(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return (lat >= 10) & (lat <= 30) & (lon > 300) & (lon <= 342.5)


def summarize_tracks(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (experiment, year), gy in df.groupby(["experiment", "year"]):
        track_count = gy["global_track_id"].nunique()
        first = gy.sort_values(["global_track_id", "time_index"]).groupby("global_track_id").head(1)
        rows.append(
            {
                "method": method,
                "experiment": experiment,
                "year": int(year),
                "n_tracks_jas_na": int(track_count),
                "n_genesis_mdr": int(point_in_mdr(first["lat"], first["lon"]).sum()),
                "n_track_points": int(len(gy)),
                "n_track_points_mdr": int(point_in_mdr(gy["lat"], gy["lon"]).sum()),
                "n_track_points_box1": int(point_in_box1(gy["lat"], gy["lon"]).sum()),
                "n_track_points_box2": int(point_in_box2(gy["lat"], gy["lon"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n: int = 3000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(123)
    means = np.array([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def experiment_summary(yearly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "n_tracks_jas_na",
        "n_genesis_mdr",
        "n_track_points",
        "n_track_points_mdr",
        "n_track_points_box1",
        "n_track_points_box2",
    ]
    for (method, experiment), g in yearly.groupby(["method", "experiment"]):
        row = {"method": method, "experiment": experiment, "n_years": len(g)}
        for metric in metrics:
            vals = g[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        rows.append(row)

    summary = pd.DataFrame(rows)
    deltas = []
    for method in summary["method"].unique():
        ctl = summary[(summary["method"] == method) & (summary["experiment"] == "CTL")]
        ggw = summary[(summary["method"] == method) & (summary["experiment"] == "GGW")]
        if ctl.empty or ggw.empty:
            continue
        ctl = ctl.iloc[0]
        ggw = ggw.iloc[0]
        row = {"method": method, "experiment": "GGW-CTL", "n_years": min(ctl["n_years"], ggw["n_years"])}
        for metric in metrics:
            cm = ctl[f"{metric}_mean"]
            gm = ggw[f"{metric}_mean"]
            row[f"{metric}_mean"] = gm - cm
            row[f"{metric}_pct"] = (gm - cm) / cm * 100.0 if cm != 0 else np.nan
        deltas.append(row)
    return pd.concat([summary, pd.DataFrame(deltas)], ignore_index=True)


def density_grid(df: pd.DataFrame, kind: str, ddeg: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if df.empty:
        lon_edges = np.arange(260, 362.5 + ddeg, ddeg)
        lat_edges = np.arange(0, 50 + ddeg, ddeg)
        return lon_edges, lat_edges, np.zeros((len(lat_edges) - 1, len(lon_edges) - 1))
    dat = df.copy()
    dat["plot_lon"] = dat["lon"].where(dat["lon"] >= 180, dat["lon"] + 360)
    if kind == "genesis":
        dat = dat.sort_values(["global_track_id", "time_index"]).groupby("global_track_id").head(1)
    lon_edges = np.arange(260, 392.5 + ddeg, ddeg)
    lat_edges = np.arange(0, 50 + ddeg, ddeg)
    h, _, _ = np.histogram2d(dat["lat"], dat["plot_lon"], bins=[lat_edges, lon_edges])
    return lon_edges, lat_edges, h


def plot_count_summary(yearly: pd.DataFrame) -> None:
    metric = "n_genesis_mdr"
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    labels = []
    means = []
    errors = []
    colors = []
    for method in ["ML", "OWZ"]:
        for exp in ["CTL", "GGW"]:
            g = yearly[(yearly["method"] == method) & (yearly["experiment"] == exp)]
            vals = g[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals)
            m = float(np.mean(vals))
            labels.append(f"{method}\n{exp}")
            means.append(m)
            errors.append([[m - lo], [hi - m]])
            colors.append("#267c8f" if method == "ML" else "#c7634d")
    x = np.arange(len(labels))
    yerr = np.array(errors).reshape(len(labels), 2).T
    ax.bar(x, means, yerr=yerr, color=colors, alpha=0.88, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MDR genesis tracks per JAS season")
    ax.set_title("ML detector versus OWZ, excluding GGW year100")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ml_vs_owz_mdr_genesis_counts.png")
    plt.close(fig)


def plot_density_panels(ml: pd.DataFrame, owz: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), dpi=160, sharex=True, sharey=True)
    for r, (method, df) in enumerate([("ML", ml), ("OWZ", owz)]):
        ctl = df[df["experiment"] == "CTL"]
        ggw = df[df["experiment"] == "GGW"]
        for c, (title, dat) in enumerate([("CTL", ctl), ("GGW", ggw), ("GGW-CTL", None)]):
            ax = axes[r, c]
            if dat is not None:
                lon_edges, lat_edges, h = density_grid(dat, "track", 5.0)
                years = dat.groupby(["experiment", "year"]).ngroups
                h = h / max(years, 1)
            else:
                lon_edges, lat_edges, h_ctl = density_grid(ctl, "track", 5.0)
                _, _, h_ggw = density_grid(ggw, "track", 5.0)
                h = h_ggw / max(ggw.groupby(["experiment", "year"]).ngroups, 1) - h_ctl / max(
                    ctl.groupby(["experiment", "year"]).ngroups, 1
                )
            vmax = np.nanmax(np.abs(h)) if title == "GGW-CTL" else np.nanmax(h)
            vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1
            cmap = "RdBu_r" if title == "GGW-CTL" else "YlOrRd"
            vmin = -vmax if title == "GGW-CTL" else 0
            mesh = ax.pcolormesh(lon_edges, lat_edges, h, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{method} {title}", fontsize=10)
            ax.set_xlim(260, 390)
            ax.set_ylim(0, 50)
            ax.grid(alpha=0.15)
            cb = fig.colorbar(mesh, ax=ax, shrink=0.82)
            cb.ax.tick_params(labelsize=7)
            if r == 1:
                ax.set_xlabel("longitude (degE; 360-390 = 0-30E)")
            if c == 0:
                ax.set_ylabel("latitude")
    fig.suptitle("JAS North Atlantic track-point density per season", y=0.99)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ml_vs_owz_track_density_panels.png")
    plt.close(fig)


def compare_ml_owz(ml_tracks: pd.DataFrame, owz_tracks: pd.DataFrame) -> None:
    yearly = pd.concat(
        [summarize_tracks(ml_tracks, "ML"), summarize_tracks(owz_tracks, "OWZ")],
        ignore_index=True,
    )
    yearly.to_csv(REPORT_DIR / "ml_vs_owz_yearly_summary.csv", index=False, encoding="utf-8-sig")
    summary = experiment_summary(yearly)
    summary.to_csv(REPORT_DIR / "ml_vs_owz_experiment_summary.csv", index=False, encoding="utf-8-sig")
    plot_count_summary(yearly)
    plot_density_panels(ml_tracks, owz_tracks)

    compact = {
        "year100_policy": "GGW year100 excluded throughout.",
        "ml_track_points": int(len(ml_tracks)),
        "owz_track_points_jas_na": int(len(owz_tracks)),
        "summary_csv": str(REPORT_DIR / "ml_vs_owz_experiment_summary.csv"),
        "yearly_csv": str(REPORT_DIR / "ml_vs_owz_yearly_summary.csv"),
        "figures": [
            str(FIG_DIR / "ml_vs_owz_mdr_genesis_counts.png"),
            str(FIG_DIR / "ml_vs_owz_track_density_panels.png"),
        ],
    }
    (REPORT_DIR / "ml_vs_owz_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] comparison -> {REPORT_DIR / 'ml_vs_owz_experiment_summary.csv'}", flush=True)


def main() -> None:
    ensure_dirs()
    copy_this_script()
    ml = apply_ml_all_years()
    owz = read_owz_all()
    compare_ml_owz(ml, owz)
    print("[DONE] CESM ML application and OWZ comparison finished.", flush=True)


if __name__ == "__main__":
    main()

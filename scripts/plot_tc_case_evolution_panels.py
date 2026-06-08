from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


ROOT = Path(os.environ.get("TC_DETECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data" / "cesm_ml_application"
YEAR_TRACK_DIR = DATA_DIR / "year_tracks"
OWZ_TRACK_FILE = DATA_DIR / "owz_track_points_jas_na_excluding_ggw100.csv"

CESM_INPUT = {
    "CTL": Path(os.environ.get("CESM_CTL_PRE", r"E:\path\to\CESM\control")),
    "GGW": Path(os.environ.get("CESM_GGW_PRE", r"G:\path\to\CESM\surface")),
}

# Change these values if you want a denser or coarser diagnosis.
PANEL_STRIDE = 4  # 1 = every 6 h, 4 = daily
N_COLS = 4
N_ROWS = 6
COMPANION_RADIUS_KM = 800.0
MAX_COMPANION_TRACKS = 2

JAS_FIRST_TIME_INDEX = 181 * 4
NO_LEAP_MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


@dataclass(frozen=True)
class CaseConfig:
    key: str
    experiment: str
    year: int
    ml_id: str | None
    owz_id: str | None
    output_name: str


CASES = [
    CaseConfig(
        key="common",
        experiment="GGW",
        year=109,
        ml_id="ML_GGW_109_0007",
        owz_id="OWZ_GGW_109_0177",
        output_name="fig09_common_case_daily_evolution.png",
    ),
    CaseConfig(
        key="ml_only",
        experiment="CTL",
        year=35,
        ml_id="ML_CTL_35_0004",
        owz_id=None,
        output_name="fig10_ml_only_case_daily_evolution.png",
    ),
    CaseConfig(
        key="owz_only",
        experiment="CTL",
        year=48,
        ml_id=None,
        owz_id="OWZ_CTL_48_0174",
        output_name="fig11_owz_only_case_daily_evolution.png",
    ),
    CaseConfig(
        key="true_ml_only",
        experiment="CTL",
        year=31,
        ml_id="ML_CTL_31_0003",
        owz_id=None,
        output_name="fig12_true_ml_only_case_daily_evolution.png",
    ),
    CaseConfig(
        key="true_owz_only",
        experiment="CTL",
        year=31,
        ml_id=None,
        owz_id="OWZ_CTL_31_0106",
        output_name="fig13_true_owz_only_case_daily_evolution.png",
    ),
]


FIRE = LinearSegmentedColormap.from_list(
    "nature_fire_extended",
    ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"],
)

ZETA_LEVELS = np.arange(-4, 12.1, 2)
RH_LEVELS = [50, 60, 70, 80]
SHEAR_LEVELS = [12.5, 20, 30]
STEERING_LEVELS_HPA = [200, 300, 500, 700, 850]


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.75,
            "axes.edgecolor": "black",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def time_label(time_index: int) -> str:
    day_of_year = time_index // 4 + 1
    hour = (time_index % 4) * 6
    remain = day_of_year
    month = 1
    for length in NO_LEAP_MONTH_LENGTHS:
        if remain <= length:
            day = remain
            break
        remain -= length
        month += 1
    else:
        month = 12
        day = 31
    return f"{month:02d}-{day:02d} {hour:02d}Z"


def jas_day_label(time_index: int) -> str:
    jas_day = (time_index - JAS_FIRST_TIME_INDEX) // 4 + 1
    return f"JAS day {jas_day}"


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lon1 = ((lon1 + 180.0) % 360.0) - 180.0
    lon2 = ((lon2 + 180.0) % 360.0) - 180.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def load_ml_tracks(experiment: str, year: int) -> pd.DataFrame:
    path = YEAR_TRACK_DIR / f"ml_tracks_{experiment}_year{year}.csv"
    df = pd.read_csv(path)
    df["time_index"] = df["time_index"].astype(int)
    df["year"] = df["year"].astype(int)
    df["lon"] = df["lon"] % 360.0
    df["plot_lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0
    return df


def load_owz_tracks(experiment: str, year: int) -> pd.DataFrame:
    df = pd.read_csv(OWZ_TRACK_FILE)
    df = df[(df["experiment"] == experiment) & (df["year"] == year)].copy()
    df["time_index"] = df["time_index"].astype(int)
    df["lon"] = df["lon"] % 360.0
    df["plot_lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0
    return df


def nearest_companion_track_ids(target: pd.DataFrame, other: pd.DataFrame) -> list[str]:
    records: list[tuple[str, float, int]] = []
    grouped = other.groupby("time_index")
    for _, row in target.iterrows():
        time_index = int(row["time_index"])
        if time_index not in grouped.groups:
            continue
        candidate = grouped.get_group(time_index)
        dist = haversine_km(float(row["lat"]), float(row["lon"]), candidate["lat"].to_numpy(), candidate["lon"].to_numpy())
        keep = dist <= COMPANION_RADIUS_KM
        if not np.any(keep):
            continue
        for gid, distance in zip(candidate.loc[keep, "global_track_id"], dist[keep]):
            records.append((str(gid), float(distance), 1))
    if not records:
        return []
    summary = (
        pd.DataFrame(records, columns=["global_track_id", "distance_km", "n"])
        .groupby("global_track_id")
        .agg(n=("n", "sum"), min_distance_km=("distance_km", "min"), mean_distance_km=("distance_km", "mean"))
        .sort_values(["n", "min_distance_km"], ascending=[False, True])
        .head(MAX_COMPANION_TRACKS)
    )
    return summary.index.tolist()


def collect_tracks(case: CaseConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    ml_all = load_ml_tracks(case.experiment, case.year)
    owz_all = load_owz_tracks(case.experiment, case.year)

    ml_ids = [case.ml_id] if case.ml_id else []
    owz_ids = [case.owz_id] if case.owz_id else []

    if case.ml_id and not case.owz_id:
        target = ml_all[ml_all["global_track_id"] == case.ml_id]
        owz_ids.extend(nearest_companion_track_ids(target, owz_all))
    if case.owz_id and not case.ml_id:
        target = owz_all[owz_all["global_track_id"] == case.owz_id]
        ml_ids.extend(nearest_companion_track_ids(target, ml_all))

    ml = ml_all[ml_all["global_track_id"].isin(ml_ids)].copy()
    owz = owz_all[owz_all["global_track_id"].isin(owz_ids)].copy()
    return ml, owz, ml_ids, owz_ids


def selected_time_indices(ml: pd.DataFrame, owz: pd.DataFrame) -> list[int]:
    all_times = pd.concat([ml["time_index"], owz["time_index"]], ignore_index=True).dropna().astype(int)
    start = int(all_times.min())
    end = int(all_times.max())
    times = set(range(start, end + 1, PANEL_STRIDE))
    for df in [ml, owz]:
        if df.empty:
            continue
        times.add(int(df["time_index"].min()))
        times.add(int(df["time_index"].max()))
    return sorted(times)


def domain_from_tracks(ml: pd.DataFrame, owz: pd.DataFrame) -> tuple[float, float, float, float]:
    tracks = pd.concat([ml[["lat", "plot_lon"]], owz[["lat", "plot_lon"]]], ignore_index=True)
    x0 = max(-110.0, math.floor(float(tracks["plot_lon"].min()) / 5.0) * 5.0 - 5.0)
    x1 = min(0.0, math.ceil(float(tracks["plot_lon"].max()) / 5.0) * 5.0 + 5.0)
    y0 = max(0.0, math.floor(float(tracks["lat"].min()) / 5.0) * 5.0 - 5.0)
    y1 = min(50.0, math.ceil(float(tracks["lat"].max()) / 5.0) * 5.0 + 5.0)
    return x0, x1, y0, y1


def level_index(ds: xr.Dataset, hpa: int) -> int:
    values = np.asarray(ds["plev"].values, dtype=float)
    return int(np.argmin(np.abs(values - hpa * 100.0)))


def subset_indices(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.where((values >= lower) & (values <= upper))[0]


def field_slice(ds: xr.Dataset, var: str, time_index: int, lat_idx: np.ndarray, lon_idx: np.ndarray, plev_idx: int | None = None) -> np.ndarray:
    arr = ds[var].isel(time=time_index, lat=lat_idx, lon=lon_idx)
    if plev_idx is not None:
        arr = arr.isel(plev=plev_idx)
    return np.asarray(arr.values, dtype=float)


def steering_wind(
    ds: xr.Dataset,
    time_index: int,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    plev_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    u = ds["U"].isel(time=time_index, plev=plev_indices, lat=lat_idx, lon=lon_idx).mean("plev")
    v = ds["V"].isel(time=time_index, plev=plev_indices, lat=lat_idx, lon=lon_idx).mean("plev")
    return smooth_field(np.asarray(u.values, dtype=float), sigma=0.7), smooth_field(np.asarray(v.values, dtype=float), sigma=0.7)


def smooth_field(field: np.ndarray, sigma: float = 0.9) -> np.ndarray:
    return gaussian_filter(field, sigma=sigma, mode="nearest")


def style_geo_axis(ax, x0: float, x1: float, y0: float, y1: float, row: int, col: int, last_row: int) -> None:
    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
    ax.set_aspect("auto")
    ax.coastlines(resolution="110m", linewidth=0.45, color="#3f3f3f")
    ax.set_facecolor("white")

    x_step = 20.0 if (x1 - x0) >= 55.0 else 10.0
    y_step = 10.0 if (y1 - y0) >= 25.0 else 5.0
    xticks = np.arange(math.ceil(x0 / x_step) * x_step, x1 + 0.1, x_step)
    yticks = np.arange(math.ceil(y0 / y_step) * y_step, y1 + 0.1, y_step)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ylabels = [f"{int(y)}N" for y in yticks]
    xlabels = [f"{abs(int(x))}W" if x < 0 else f"{int(x)}E" for x in xticks]
    ax.set_xticklabels(xlabels if row == last_row else [])
    ax.set_yticklabels(ylabels if col == 0 else [])
    ax.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="out",
        length=2.4,
        width=0.75,
        color="black",
        labelcolor="black",
        pad=1.5,
    )
    ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False, length=0, width=0)
    ax.grid(True, which="major", color="#b8b8b8", linestyle=(0, (3, 3)), linewidth=0.5, alpha=0.75)
    ax.grid(False, which="minor")
    for side in ["left", "right", "bottom", "top"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.75)
    if row == last_row:
        ax.set_xlabel("Longitude")
    else:
        ax.set_xlabel("")
    if col == 0:
        ax.set_ylabel("Latitude")
    else:
        ax.set_ylabel("")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontweight("bold")
        label.set_fontsize(8)


def status_text(time_index: int, ml: pd.DataFrame, owz: pd.DataFrame) -> str:
    ml_on = "ML on" if not ml[ml["time_index"] == time_index].empty else "ML off"
    owz_on = "OWZ on" if not owz[owz["time_index"] == time_index].empty else "OWZ off"
    return f"{jas_day_label(time_index)}\n{time_label(time_index)}\n{ml_on}; {owz_on}"


def plot_tracks_for_time(ax, ml: pd.DataFrame, owz: pd.DataFrame, time_index: int) -> None:
    transform = ccrs.PlateCarree()
    if not ml.empty:
        for _, track in ml.groupby("global_track_id"):
            role_alpha = 0.82 if str(track["global_track_id"].iloc[0]).startswith("ML_") else 0.55
            ax.plot(track["plot_lon"], track["lat"], color="#e84a35", linewidth=1.05, alpha=0.42, transform=transform, zorder=4)
            now = track[track["time_index"] == time_index]
            if not now.empty:
                ax.scatter(
                    now["plot_lon"],
                    now["lat"],
                    s=35,
                    marker="o",
                    facecolor="#e84a35",
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=role_alpha,
                    transform=transform,
                    zorder=8,
                )
    if not owz.empty:
        for _, track in owz.groupby("global_track_id"):
            ax.plot(track["plot_lon"], track["lat"], color="#41b6c4", linewidth=1.05, alpha=0.48, transform=transform, zorder=5)
            now = track[track["time_index"] == time_index]
            if not now.empty:
                ax.scatter(
                    now["plot_lon"],
                    now["lat"],
                    s=40,
                    marker="^",
                    facecolor="#41b6c4",
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.88,
                    transform=transform,
                    zorder=9,
                )


def plot_wind_arrows(ax, lon2d: np.ndarray, lat2d: np.ndarray, u: np.ndarray, v: np.ndarray, add_key: bool = False):
    y_step = max(1, len(lat2d[:, 0]) // 7)
    x_step = max(1, len(lon2d[0, :]) // 7)
    qv = ax.quiver(
        lon2d[::y_step, ::x_step],
        lat2d[::y_step, ::x_step],
        u[::y_step, ::x_step],
        v[::y_step, ::x_step],
        transform=ccrs.PlateCarree(),
        color="#1f1f1f",
        scale=95,
        width=0.0024,
        headwidth=3.7,
        headlength=4.8,
        headaxislength=4.2,
        pivot="middle",
        alpha=0.86,
        zorder=6,
    )
    if add_key:
        key = ax.quiverkey(
            qv,
            X=0.82,
            Y=0.91,
            U=10,
            label="10 m s^-1",
            labelpos="W",
            coordinates="axes",
            color="#1f1f1f",
            labelcolor="black",
            fontproperties={"family": "Times New Roman", "weight": "bold", "size": 7},
        )
        return key
    return None


def add_start_end_markers(ax, ml: pd.DataFrame, owz: pd.DataFrame) -> None:
    transform = ccrs.PlateCarree()
    for df, color, marker in [(ml, "#e84a35", "o"), (owz, "#41b6c4", "^")]:
        if df.empty:
            continue
        for _, track in df.groupby("global_track_id"):
            first = track.sort_values("time_index").iloc[0]
            last = track.sort_values("time_index").iloc[-1]
            ax.scatter(first["plot_lon"], first["lat"], s=18, marker=marker, facecolor="white", edgecolor=color, linewidth=0.8, transform=transform, zorder=10)
            ax.scatter(last["plot_lon"], last["lat"], s=24, marker=marker, facecolor=color, edgecolor="black", linewidth=0.35, transform=transform, zorder=10)


def track_windows(case: CaseConfig, ml: pd.DataFrame, owz: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, df in [("ML", ml), ("OWZ", owz)]:
        if df.empty:
            continue
        for gid, track in df.groupby("global_track_id"):
            start = int(track["time_index"].min())
            end = int(track["time_index"].max())
            role = "target"
            if method == "ML" and case.ml_id is not None and gid != case.ml_id:
                role = "nearby_companion"
            if method == "OWZ" and case.owz_id is not None and gid != case.owz_id:
                role = "nearby_companion"
            if method == "ML" and case.ml_id is None:
                role = "nearby_companion"
            if method == "OWZ" and case.owz_id is None:
                role = "nearby_companion"
            rows.append(
                {
                    "case": case.key,
                    "experiment": case.experiment,
                    "year": case.year,
                    "method": method,
                    "track_id": gid,
                    "role": role,
                    "start_time_index": start,
                    "end_time_index": end,
                    "start_label": time_label(start),
                    "end_label": time_label(end),
                    "n_points": int(len(track)),
                }
            )
    return rows


def plot_case(case: CaseConfig) -> list[dict[str, object]]:
    ml, owz, ml_ids, owz_ids = collect_tracks(case)
    if ml.empty and owz.empty:
        raise ValueError(f"No track points found for {case.key}")

    times = selected_time_indices(ml, owz)
    if len(times) > N_COLS * N_ROWS:
        times = times[: N_COLS * N_ROWS]
    n_rows = int(math.ceil(len(times) / N_COLS))

    x0, x1, y0, y1 = domain_from_tracks(ml, owz)
    ds_path = CESM_INPUT[case.experiment] / f"cesm_{case.year}.nc"
    ds = xr.open_dataset(ds_path, decode_times=False)
    lat_values = np.asarray(ds["lat"].values, dtype=float)
    lon_values = np.asarray(ds["lon"].values, dtype=float)
    lon_plot_values = ((lon_values + 180.0) % 360.0) - 180.0
    lat_idx = subset_indices(lat_values, y0, y1)
    lon_idx = subset_indices(lon_plot_values, x0, x1)
    lats = lat_values[lat_idx]
    lons = lon_plot_values[lon_idx]
    lon2d, lat2d = np.meshgrid(lons, lats)
    zeta_norm = BoundaryNorm(ZETA_LEVELS, FIRE.N)
    plev850 = level_index(ds, 850)
    plev700 = level_index(ds, 700)
    steering_plev = [level_index(ds, hpa) for hpa in STEERING_LEVELS_HPA]

    fig = plt.figure(figsize=(13.6, 2.45 * n_rows + 1.10), dpi=420)
    axes = []
    for i in range(n_rows * N_COLS):
        ax = fig.add_subplot(n_rows, N_COLS, i + 1, projection=ccrs.PlateCarree())
        axes.append(ax)
        if i >= len(times):
            ax.set_axis_off()
            continue
        time_index = int(times[i])
        zeta850 = smooth_field(field_slice(ds, "zeta", time_index, lat_idx, lon_idx, plev850) * 1e5)
        rh700 = smooth_field(field_slice(ds, "RH", time_index, lat_idx, lon_idx, plev700))
        shear = smooth_field(field_slice(ds, "Wshear_850_200", time_index, lat_idx, lon_idx))
        steer_u, steer_v = steering_wind(ds, time_index, lat_idx, lon_idx, steering_plev)
        row, col = divmod(i, N_COLS)
        last_row = (len(times) - 1) // N_COLS
        style_geo_axis(ax, x0, x1, y0, y1, row, col, last_row)
        cf = ax.contourf(
            lon2d,
            lat2d,
            zeta850,
            levels=ZETA_LEVELS,
            cmap=FIRE,
            norm=zeta_norm,
            extend="both",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        rh_cs = ax.contour(lon2d, lat2d, rh700, levels=RH_LEVELS, colors="#1a9850", linewidths=0.75, transform=ccrs.PlateCarree(), zorder=3)
        sh_cs = ax.contour(
            lon2d,
            lat2d,
            shear,
            levels=SHEAR_LEVELS,
            colors="#6a3d9a",
            linewidths=0.70,
            linestyles="--",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        plot_wind_arrows(ax, lon2d, lat2d, steer_u, steer_v, add_key=(i == 0))
        plot_tracks_for_time(ax, ml, owz, time_index)
        add_start_end_markers(ax, ml, owz)
        ax.text(
            0.02,
            0.98,
            status_text(time_index, ml, owz),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            fontfamily="Times New Roman",
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4},
            zorder=20,
        )
        if i == 0:
            if not ml.empty:
                ax.plot([], [], color="#e84a35", marker="o", linewidth=1.1, markersize=4.2, label="ML")
            if not owz.empty:
                ax.plot([], [], color="#41b6c4", marker="^", linewidth=1.1, markersize=4.5, label="OWZ")
            ax.plot([], [], color="#1a9850", linewidth=0.8, label="RH700")
            ax.plot([], [], color="#6a3d9a", linestyle="--", linewidth=0.8, label="Shear")
            leg = ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="white", framealpha=1.0)
            leg.get_frame().set_linewidth(0.0)
            for text in leg.get_texts():
                text.set_fontfamily("Times New Roman")
                text.set_fontweight("bold")
                text.set_fontsize(8)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.985, bottom=0.075, wspace=0.08, hspace=0.16)
    cax = fig.add_axes([0.25, 0.03, 0.50, 0.014])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("zeta850 (1e-5 s^-1)", fontfamily="Times New Roman", fontweight="bold", fontsize=8, labelpad=2)
    cb.outline.set_linewidth(0.75)
    cb.ax.tick_params(direction="out", length=2.4, width=0.75, labelsize=8)
    for label in cb.ax.get_xticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontweight("bold")

    output = FIG_DIR / case.output_name
    fig.savefig(output, dpi=420, facecolor="white")
    plt.close(fig)
    ds.close()

    rows = track_windows(case, ml, owz)
    for row in rows:
        row["ml_track_ids_plotted"] = ";".join(ml_ids)
        row["owz_track_ids_plotted"] = ";".join(owz_ids)
        row["panel_stride_time_steps"] = PANEL_STRIDE
        row["output_figure"] = str(output)
    return rows


def main() -> None:
    set_nature_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for case in CASES:
        all_rows.extend(plot_case(case))
    summary = pd.DataFrame(all_rows)
    summary.to_csv(FIG_DIR / "case_evolution_track_windows.csv", index=False)


if __name__ == "__main__":
    main()

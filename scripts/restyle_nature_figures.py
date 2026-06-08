from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


ROOT = Path(os.environ.get("TC_DETECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORT_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
SCRIPT_DIR = ROOT / "scripts"
BACKUP_DIR = FIG_DIR / "original_style_backup"


@dataclass
class AxisTicks:
    x_min: float | None = None
    x_max: float | None = None
    x_major: float | None = None
    x_minor_count: int = 0
    y_min: float | None = None
    y_max: float | None = None
    y_major: float | None = None
    y_minor_count: int = 0


@dataclass
class NatureStyle:
    figsize: tuple[float, float] = (7.2, 4.8)
    dpi: int = 600
    font_family: str = "Times New Roman"
    font_size: float = 10.0
    spine_width: float = 0.75
    grid_width: float = 0.5
    tick_width: float = 0.75
    tick_length: float = 3.0
    pad: float = 2.0


STYLE = NatureStyle()


# Edit these dictionaries when you want custom tick ranges/spacing.
TICKS = {
    "detection": AxisTicks(y_min=0.94, y_max=1.005, y_major=0.02, y_minor_count=1),
    "svr": AxisTicks(x_min=6, x_max=24, x_major=6, y_min=0, y_max=220, y_major=40, y_minor_count=1),
    "counts": AxisTicks(y_min=0, y_max=10.5, y_major=2, y_minor_count=1),
    "density": AxisTicks(x_min=260, x_max=390, x_major=30, y_min=0, y_max=50, y_major=10, x_minor_count=2, y_minor_count=1),
}


COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "nature_fire",
    ["#fff7bc", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04"],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "nature_balance",
    ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
)


def set_rcparams() -> None:
    mpl.rcParams.update(
        {
            "font.family": STYLE.font_family,
            "font.size": STYLE.font_size,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": STYLE.spine_width,
            "axes.edgecolor": "black",
            "xtick.labelsize": STYLE.font_size,
            "ytick.labelsize": STYLE.font_size,
            "legend.fontsize": STYLE.font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": STYLE.dpi,
            "figure.dpi": STYLE.dpi,
        }
    )


def make_bold_text(ax: plt.Axes) -> None:
    text_objs = [ax.xaxis.label, ax.yaxis.label]
    text_objs.extend(ax.get_xticklabels())
    text_objs.extend(ax.get_yticklabels())
    title = ax.get_title()
    if title:
        ax.set_title(title, fontweight="bold", fontsize=STYLE.font_size)
    for obj in text_objs:
        obj.set_fontfamily(STYLE.font_family)
        obj.set_fontweight("bold")
        obj.set_fontsize(STYLE.font_size)


def style_axis(ax: plt.Axes, ticks: AxisTicks | None = None, spine_color: str = "black") -> None:
    for side in ["left", "right", "bottom", "top"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(spine_color)
        ax.spines[side].set_linewidth(STYLE.spine_width)

    ax.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="out",
        length=STYLE.tick_length,
        width=STYLE.tick_width,
        color=spine_color,
        labelcolor="black",
        pad=STYLE.pad,
    )
    ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False, length=0, width=0)

    ax.grid(True, which="major", color="#b8b8b8", linestyle=(0, (3, 3)), linewidth=STYLE.grid_width, alpha=0.75)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)

    if ticks is not None:
        if ticks.x_min is not None or ticks.x_max is not None:
            ax.set_xlim(ticks.x_min, ticks.x_max)
        if ticks.y_min is not None or ticks.y_max is not None:
            ax.set_ylim(ticks.y_min, ticks.y_max)
        if ticks.x_major is not None:
            ax.xaxis.set_major_locator(MultipleLocator(ticks.x_major))
        if ticks.y_major is not None:
            ax.yaxis.set_major_locator(MultipleLocator(ticks.y_major))
        if ticks.x_minor_count and ticks.x_minor_count > 0:
            ax.xaxis.set_minor_locator(AutoMinorLocator(ticks.x_minor_count + 1))
        if ticks.y_minor_count and ticks.y_minor_count > 0:
            ax.yaxis.set_minor_locator(AutoMinorLocator(ticks.y_minor_count + 1))

    make_bold_text(ax)


def style_legend(ax: plt.Axes, **kwargs) -> None:
    leg = ax.legend(frameon=True, facecolor="white", framealpha=1.0, edgecolor="white", **kwargs)
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_linewidth(0.0)
    frame.set_facecolor("white")
    frame.set_edgecolor("white")
    for text in leg.get_texts():
        text.set_fontfamily(STYLE.font_family)
        text.set_fontweight("bold")
        text.set_fontsize(STYLE.font_size)


def style_colorbar(cb) -> None:
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(STYLE.spine_width)
    cb.ax.tick_params(
        which="major",
        direction="out",
        length=STYLE.tick_length,
        width=STYLE.tick_width,
        color="black",
        labelcolor="black",
        labelsize=STYLE.font_size,
    )
    cb.ax.tick_params(which="minor", length=0, width=0)
    for label in cb.ax.get_yticklabels() + cb.ax.get_xticklabels():
        label.set_fontfamily(STYLE.font_family)
        label.set_fontweight("bold")
        label.set_fontsize(STYLE.font_size)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIG_DIR / filename, dpi=STYLE.dpi, facecolor="white")
    plt.close(fig)


def backup_existing_pngs() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for png in FIG_DIR.glob("*.png"):
        target = BACKUP_DIR / png.name
        if not target.exists():
            shutil.copy2(png, target)


def plot_detection_metrics() -> None:
    df = pd.read_csv(REPORT_DIR / "detection_model_metrics.csv")
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=STYLE.figsize)
    ax.bar(x - width / 2, df["f1"], width=width, color=COLORS["blue"], label="F1", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, df["roc_auc"], width=width, color=COLORS["orange"], label="ROC-AUC", edgecolor="black", linewidth=0.4)
    labels = [m.replace("_", "-").replace("svc", "SVC").replace("logistic", "Logistic").replace("random-forest", "RF") for m in df["model"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Score")
    style_axis(ax, TICKS["detection"])
    style_legend(ax, loc="lower right", ncol=1)
    fig.tight_layout()
    save_figure(fig, "detection_model_metrics.png")


def plot_svr_path_metrics() -> None:
    df = pd.read_csv(REPORT_DIR / "svr_path_metrics.csv")
    order = [
        ("linear_regression", "linear", "Linear regression", COLORS["black"], "s"),
        ("persistence_motion", "baseline", "Persistence", COLORS["orange"], "D"),
        ("SVR", "linear", "SVR-linear", COLORS["blue"], "o"),
        ("SVR", "rbf", "SVR-RBF", COLORS["red"], "^"),
        ("SVR", "poly", "SVR-poly", COLORS["green"], "v"),
        ("SVR", "sigmoid", "SVR-sigmoid", COLORS["purple"], "P"),
    ]
    fig, ax = plt.subplots(figsize=STYLE.figsize)
    for model, kernel, label, color, marker in order:
        g = df[(df["model"] == model) & (df["kernel"] == kernel)].sort_values("lead_h")
        if g.empty:
            continue
        ax.plot(
            g["lead_h"],
            g["ate_mean_km"],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.6,
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=label,
        )
    ax.set_xlabel("Forecast lead (h)")
    ax.set_ylabel("Mean ATE (km)")
    style_axis(ax, TICKS["svr"])
    style_legend(ax, loc="upper left", ncol=2)
    fig.tight_layout()
    save_figure(fig, "svr_path_ate_by_lead.png")


def asymmetric_errors(mean: float, low: float, high: float) -> tuple[float, float]:
    return mean - low, high - mean


def plot_mdr_genesis_counts() -> None:
    df = pd.read_csv(REPORT_DIR / "ml_vs_owz_experiment_summary.csv")
    df = df[df["experiment"].isin(["CTL", "GGW"])].copy()
    rows = []
    for method in ["ML", "OWZ"]:
        for exp in ["CTL", "GGW"]:
            r = df[(df["method"] == method) & (df["experiment"] == exp)].iloc[0]
            mean = r["n_genesis_mdr_mean"]
            low = r["n_genesis_mdr_ci95_low"]
            high = r["n_genesis_mdr_ci95_high"]
            rows.append((f"{method}\n{exp}", method, exp, mean, *asymmetric_errors(mean, low, high)))
    labels, methods, exps, means, err_low, err_high = zip(*rows)
    colors = [COLORS["blue"] if m == "ML" else COLORS["red"] for m in methods]
    fig, ax = plt.subplots(figsize=STYLE.figsize)
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=np.vstack([err_low, err_high]),
        color=colors,
        edgecolor="black",
        linewidth=0.45,
        capsize=3,
        error_kw={"elinewidth": 0.75, "capthick": 0.75, "ecolor": "black"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MDR genesis tracks per JAS")
    style_axis(ax, TICKS["counts"])
    fig.tight_layout()
    save_figure(fig, "ml_vs_owz_mdr_genesis_counts.png")


def density_grid(df: pd.DataFrame, ddeg: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dat = df.copy()
    dat["plot_lon"] = dat["lon"].where(dat["lon"] >= 180, dat["lon"] + 360)
    lon_edges = np.arange(260, 392.5 + ddeg, ddeg)
    lat_edges = np.arange(0, 50 + ddeg, ddeg)
    h, _, _ = np.histogram2d(dat["lat"], dat["plot_lon"], bins=[lat_edges, lon_edges])
    return lon_edges, lat_edges, h


def plot_density_panels() -> None:
    ml = pd.read_csv(DATA_DIR / "cesm_ml_application" / "ml_detected_track_points.csv")
    owz = pd.read_csv(DATA_DIR / "cesm_ml_application" / "owz_track_points_jas_na_excluding_ggw100.csv")
    fig, axes = plt.subplots(2, 3, figsize=STYLE.figsize, sharex=True, sharey=True)
    panel_info = [("ML", ml), ("OWZ", owz)]
    col_info = ["CTL", "GGW", "GGW-CTL"]
    for r, (method, df) in enumerate(panel_info):
        ctl = df[df["experiment"] == "CTL"]
        ggw = df[df["experiment"] == "GGW"]
        years_ctl = ctl.groupby(["experiment", "year"]).ngroups
        years_ggw = ggw.groupby(["experiment", "year"]).ngroups
        lon_edges, lat_edges, h_ctl = density_grid(ctl)
        _, _, h_ggw = density_grid(ggw)
        arrays = [h_ctl / years_ctl, h_ggw / years_ggw, h_ggw / years_ggw - h_ctl / years_ctl]
        for c, label in enumerate(col_info):
            ax = axes[r, c]
            arr = arrays[c]
            if label == "GGW-CTL":
                vmax = np.nanpercentile(np.abs(arr), 98)
                vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1
                mesh = ax.pcolormesh(lon_edges, lat_edges, arr, shading="auto", cmap=DIVERGING, vmin=-vmax, vmax=vmax)
            else:
                vmax = np.nanpercentile(arr, 98)
                vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1
                mesh = ax.pcolormesh(lon_edges, lat_edges, arr, shading="auto", cmap=SEQUENTIAL, vmin=0, vmax=vmax)
            ax.text(
                0.03,
                0.93,
                f"{method} {label}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=STYLE.font_size,
                fontfamily=STYLE.font_family,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 1.5},
            )
            if r == 1:
                ax.set_xlabel("Longitude (degE)")
            if c == 0:
                ax.set_ylabel("Latitude")
            style_axis(ax, TICKS["density"])
            cb = fig.colorbar(mesh, ax=ax, shrink=0.74, pad=0.02)
            style_colorbar(cb)
    fig.tight_layout(w_pad=0.6, h_pad=0.9)
    save_figure(fig, "ml_vs_owz_track_density_panels.png")


def main() -> None:
    set_rcparams()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    backup_existing_pngs()
    plot_detection_metrics()
    plot_svr_path_metrics()
    plot_mdr_genesis_counts()
    plot_density_panels()
    shutil.copy2(Path(__file__).resolve(), SCRIPT_DIR / Path(__file__).name)


if __name__ == "__main__":
    main()

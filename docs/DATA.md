# Data Notes

## Included CSV Files

The repository includes processed CSV files that were generated from IBTrACS, ERA5 and CESM-derived environmental fields.

Main files:

- `data/training_detection_samples.csv`: final detection samples used for model fitting.
- `data/obs_detection_samples_features.csv`: observation-derived TC/background feature table.
- `data/svr_path_samples.csv`: short-term track-prediction samples.
- `data/cesm_ctl_background_samples.csv`: CESM CTL background negatives used for training transfer robustness.
- `data/ibtracs_na_jas_1991_2020_track_points.csv`: filtered IBTrACS North Atlantic JAS track points.
- `data/cesm_ml_application/ml_detected_track_points.csv`: ML-detected CESM track points.
- `data/cesm_ml_application/owz_track_points_jas_na_excluding_ggw100.csv`: OWZ track points used in the ML-vs-OWZ comparison.
- `data/cesm_ml_application/year_tracks/`: yearly ML track-point tables.

## Excluded Raw Data

The following raw datasets are not committed:

- IBTrACS raw NetCDF.
- ERA5 daily or subdaily raw fields.
- CESM CTL/GGW raw or preprocessed NetCDF output.
- OWZ intermediate NetCDF outputs.

To rerun the full pipeline, configure these environment variables:

```bash
TC_DETECT_ROOT=/path/to/repo
IBTRACS_PATH=/path/to/IBTrACS.ALL.v04r01.nc
ERA5_ALL_PATH=/path/to/all_era5.nc
CESM_CTL_PRE=/path/to/CESM/control
CESM_GGW_PRE=/path/to/CESM/surface
OWZ_CTL=/path/to/OWZ/control
OWZ_GGW=/path/to/OWZ/surface
```

## 中文说明

仓库中上传的是已经筛选和处理后的 CSV 表格，不包含原始 NetCDF 数据。若需要完整复现实验，需要自行准备 IBTrACS、ERA5、CESM CTL/GGW 输出以及 OWZ 预处理结果，并通过环境变量指定路径。


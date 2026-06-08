# ML-TC-Detection-CESM

Machine-learning tropical cyclone detection and short-term track prediction in CESM, with an independent comparison against the physically based OWZ detector.

机器学习热带气旋检测与短期路径预测项目。该仓库基于 IBTrACS、ERA5 和 CESM 输出资料构建热带气旋检测器，并与传统 OWZ 方法进行对比。

## What This Repository Contains

This repository contains the public, reproducible artifacts from a course-project experiment:

- Python scripts for sample construction, model training, CESM application, figure styling and case diagnosis.
- Trained detection and SVR track-prediction models in `models/`.
- Processed CSV samples and result tables in `data/` and `reports/`.
- Main figures used in the report and defense slides in `figures/`.

Raw CESM, ERA5 and IBTrACS NetCDF files are not included because they are large and may have separate data-use terms. The scripts can be rerun when those data paths are provided through environment variables.

本仓库包含课程项目中的可公开产物：训练脚本、训练好的模型、筛选后的 CSV 样本、统计结果表和主要论文/答辩图。原始 CESM、ERA5 和 IBTrACS NetCDF 数据不上传。

## Scientific Workflow

1. **Observation-based samples**
   - Positive samples: IBTrACS North Atlantic tropical cyclones, 1991-2020, JAS, 6-hourly.
   - Negative samples: ERA5 background points at least 1000 km away from IBTrACS TC centers.
   - Additional CESM CTL weak-OWZ background samples are used only for training transfer robustness.

2. **Feature engineering**
   - Position and season: latitude, sine/cosine longitude, sine/cosine month.
   - Multi-level state: U, V, T, q and RH at 200, 300, 500, 700 and 850 hPa.
   - Dynamics: 850/500 hPa vorticity, 850-200 hPa vertical wind shear.
   - Steering flow: layer-mean U/V over 200-850 hPa.
   - Local neighborhood predictors: vorticity maxima, RH mean and minimum shear within a local radius.

3. **TC/background detection models**
   - Logistic Regression
   - Linear-SVC
   - RBF-SVC
   - Random Forest

4. **Short-term path prediction**
   - Linear regression and SVR models predict future displacement at 6, 12, 18 and 24 h.
   - The target is zonal/meridional displacement, not raw longitude/latitude.
   - Persistence-motion forecasts are used as a baseline.

5. **CESM transfer and OWZ comparison**
   - The best detection model is transferred to CESM CTL/GGW experiments.
   - ML tracks are compared with OWZ tracks for genesis counts, track density and case-level behavior.

## Main Results

Detection skill on the independent observation test period:

| Model | F1 | ROC-AUC |
|---|---:|---:|
| RBF-SVC | 0.976 | 0.996 |
| Linear-SVC | 0.975 | 0.995 |
| Logistic Regression | 0.975 | 0.995 |
| Random Forest | 0.969 | 0.995 |

Path prediction:

- Linear SVR is the most robust short-term path model.
- Linear-SVR 24 h mean ATE is approximately 155.8 km.
- More complex SVR kernels did not outperform the linear model.

CESM/OWZ comparison:

- ML and OWZ both indicate stronger MDR genesis under the GGW experiment.
- Case diagnostics show that ML follows vorticity-centered, sample-like TC structure, while OWZ follows a threshold-qualified environment center.
- OWZ can lag behind the vorticity maximum because its center combines an OWZ-weighted cluster center and a minimum-shear location.

## Repository Structure

```text
ML-TC-Detection-CESM/
  data/                         Processed CSV samples and CESM/OWZ application tables
  figures/                      Main figures and case-diagnosis plots
  models/                       Trained joblib models
  reports/                      Metrics, summaries and experiment reports
  scripts/                      Training, application and plotting scripts
  docs/                         Data/model notes
  requirements.txt              Python dependency list
```

## Quick Start

Create an environment:

```bash
conda create -n ml-tc-detect python=3.10
conda activate ml-tc-detect
pip install -r requirements.txt
```

Use the released CSVs, reports and figures:

```bash
python scripts/restyle_nature_figures.py
```

Rerun the full training pipeline with raw data:

```bash
export TC_DETECT_ROOT=/path/to/ML-TC-Detection-CESM
export IBTRACS_PATH=/path/to/IBTrACS.ALL.v04r01.nc
export ERA5_ALL_PATH=/path/to/all_era5.nc
export CESM_CTL_PRE=/path/to/CESM/control
export CESM_GGW_PRE=/path/to/CESM/surface
export OWZ_CTL=/path/to/OWZ/control
export OWZ_GGW=/path/to/OWZ/surface

python scripts/run_training_pipeline.py
python scripts/apply_cesm_ml_detector.py
python scripts/plot_tc_case_evolution_panels.py
```

On Windows PowerShell:

```powershell
$env:TC_DETECT_ROOT="D:\path\to\ML-TC-Detection-CESM"
$env:IBTRACS_PATH="D:\path\to\IBTrACS.ALL.v04r01.nc"
$env:ERA5_ALL_PATH="E:\path\to\all_era5.nc"
$env:CESM_CTL_PRE="E:\path\to\CESM\control"
$env:CESM_GGW_PRE="G:\path\to\CESM\surface"
$env:OWZ_CTL="E:\path\to\OWZ\control"
$env:OWZ_GGW="G:\path\to\OWZ\surface"

python scripts\run_training_pipeline.py
```

## 中文说明

本项目的目标是构建一个可以应用于 CESM 模式输出的热带气旋检测器，并与传统 OWZ 方法进行对比。机器学习模型并不是学习 OWZ 标签，而是使用 IBTrACS 正样本和 ERA5/CESM 背景负样本训练 TC/background 二分类器。随后将最优模型应用到 CESM CTL 和 GGW 试验中，比较 MDR 生成数、轨迹密度和典型个例。

核心结论：

- RBF-SVC 在观测独立测试集上取得最高检测性能，F1 约为 0.976，ROC-AUC 约为 0.996。
- 线性 SVR 是最稳定的短期路径预测模型，24 h 平均绝对路径误差约为 155.8 km。
- ML 与 OWZ 均显示 GGW 试验中 MDR 生成数增强。
- 个例分析表明，ML 更接近涡度中心和观测样本相似性，而 OWZ 更接近满足物理阈值的环境团簇中心，因此 OWZ 中心可能相对涡度最大值滞后。

## Data Policy

Included:

- Processed CSV samples.
- Trained joblib models.
- Summary metrics and figures.

Not included:

- Raw CESM NetCDF files.
- Raw ERA5 files.
- Raw IBTrACS NetCDF file.

Please download or request the raw datasets from their original providers before rerunning the full pipeline.

## License

No formal open-source license is attached yet. Please contact the author before reusing the code, models or processed data outside academic review or course-project evaluation.


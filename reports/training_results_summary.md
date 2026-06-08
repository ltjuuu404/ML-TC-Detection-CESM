# TC Detect + SVR 训练结果摘要

输出目录：`G:\machine_learning\tc_detect_svr`

## year100处理

GGW 的 `year100` 已从 CESM 年份清单和后续 OWZ 对比口径中排除。当前清单文件为：

`G:\machine_learning\tc_detect_svr\data\included_cesm_years_excluding_ggw100.csv`

## 筛选后的数据

| 数据文件 | 样本量 | 说明 |
|---|---:|---|
| `ibtracs_na_jas_1991_2020_track_points.csv` | 8751 | IBTrACS北大西洋JAS轨迹点 |
| `obs_detection_samples_features.csv` | 13251 | IBTrACS/ERA5检测样本，8751正样本 + 4500负样本 |
| `cesm_ctl_background_samples.csv` | 700 | CESM CTL明确背景负样本 |
| `training_detection_samples.csv` | 13951 | 检测模型训练总表 |
| `svr_path_samples.csv` | 28878 | 6/12/18/24小时路径预测样本 |

## 检测模型结果

测试集为 2016-2020 年 IBTrACS/ERA5 观测样本。最佳模型为 RBF-SVC。

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVC-RBF | 0.967 | 0.986 | 0.967 | 0.976 | 0.996 |
| SVC-linear | 0.966 | 0.981 | 0.969 | 0.975 | 0.995 |
| Logistic | 0.965 | 0.981 | 0.968 | 0.975 | 0.995 |
| Random Forest | 0.958 | 0.989 | 0.950 | 0.969 | 0.995 |

模型文件：

`G:\machine_learning\tc_detect_svr\models\detection_best.joblib`

## SVR路径预测结果

ATE 为平均绝对路径误差，单位 km。线性模型在当前特征工程下最稳，RBF/多项式核没有超过线性基线。

| 预报时效 | 最佳SVR核函数 | SVR平均ATE | 线性回归平均ATE | 持久性平均ATE |
|---:|---|---:|---:|---:|
| 6 h | linear | 25.2 | 25.1 | 26.1 |
| 12 h | linear | 61.0 | 60.6 | 65.6 |
| 18 h | linear | 104.9 | 103.8 | 115.0 |
| 24 h | linear | 155.8 | 154.0 | 171.8 |

## 图件

| 图件 | 内容 |
|---|---|
| `figures\detection_model_metrics.png` | 检测模型 F1 与 ROC-AUC |
| `figures\svr_path_ate_by_lead.png` | 不同SVR核函数与基线的ATE随时效变化 |

## 方法说明

主检测模型不使用 `OWZ` 作为输入特征，避免把传统方法规则直接喂给机器学习模型。输入特征主要为 `U/V/T/Q/RH`、850/500 hPa相对涡度、850-200 hPa垂直风切变、深层引导流和500 km邻域统计量。OWZ结果保留为后续独立比较基准。

ERA5当前为 daily 资料，因此本版训练中环境场代表日尺度背景；IBTrACS路径点仍使用6小时时间步。报告中应明确这一限制。若后续补充6小时ERA5，可直接复用当前脚本结构重建特征表。

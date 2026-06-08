# CESM 全场应用与 OWZ 对比结果摘要

输出目录：`G:\machine_learning\tc_detect_svr`

## 1. 这一步做了什么

本阶段把前面训练好的最佳检测模型 `detection_best.joblib` 应用到 CESM CTL 与 GGW 的 JAS 北大西洋区域，并与已有 OWZ 检测结果进行同口径比较。

继续执行的约束：

1. GGW `year100` 全部排除。
2. 原始 CESM、ERA5、IBTrACS 与 OWZ 数据只读。
3. 派生轨迹、表格、图件写入 `G:\machine_learning\tc_detect_svr`。

## 2. ML 检测流程

对每个 CESM 年文件：

1. 只读取 JAS、北大西洋区域、所需气压层和变量。
2. 用宽松物理条件找候选中心：
   - 850 hPa 涡度为局地极大值；
   - `zeta850 > 1.0e-5 s^-1`；
   - `zeta500 > 0`；
   - `850-200 hPa` 风切变 `< 30 m s^-1`；
   - `RH700 > 35%`。
3. 每个时次最多保留24个动力扰动最强候选。
4. 对候选点提取与训练阶段一致的特征，包括多层风温湿、850/500 hPa涡度、垂直风切变、深层引导流和500 km邻域统计量。
5. 用最佳 ML 检测模型输出 `prob_tc`。
6. 保留 `prob_tc >= 0.55` 的候选点。
7. 同一时次 800 km 内合并重复中心。
8. 相邻6小时时次按 800 km 最大距离连接成轨迹。
9. 轨迹持续时间需至少8个6小时步长，即48小时。

## 3. 生成的数据文件

| 文件 | 内容 |
|---|---|
| `data/cesm_ml_application/ml_detected_track_points.csv` | ML检测出的CESM轨迹点 |
| `data/cesm_ml_application/owz_track_points_jas_na_excluding_ggw100.csv` | 同口径筛选后的OWZ轨迹点 |
| `data/cesm_ml_application/year_tracks/` | 59个逐年ML轨迹缓存 |
| `reports/ml_vs_owz_yearly_summary.csv` | 逐年统计 |
| `reports/ml_vs_owz_experiment_summary.csv` | CTL/GGW/差值汇总 |
| `figures/ml_vs_owz_mdr_genesis_counts.png` | MDR生成数对比柱状图 |
| `figures/ml_vs_owz_track_density_panels.png` | CTL/GGW/GGW-CTL路径密度对比图 |

## 4. 样本规模

| 方法 | CTL年份 | GGW年份 | 轨迹点数 |
|---|---:|---:|---:|
| ML | 30 | 29 | 15099 |
| OWZ | 30 | 29 | 26275 |

GGW为29年是因为 `year100` 已按要求排除。

## 5. 主要结果

### 5.1 JAS北大西洋轨迹数量

| 方法 | CTL | GGW | GGW-CTL | 相对变化 |
|---|---:|---:|---:|---:|
| ML | 13.83 | 16.31 | +2.48 | +17.9% |
| OWZ | 31.33 | 32.97 | +1.63 | +5.2% |

ML检测到的绝对轨迹数少于OWZ，说明当前ML阈值更保守；但ML仍然检测到GGW后TC活动增强。

### 5.2 MDR生成数量

MDR定义为 `10N-20N, 70W-17.5W`。

| 方法 | CTL | GGW | GGW-CTL | 相对变化 |
|---|---:|---:|---:|---:|
| ML | 6.87 | 8.45 | +1.58 | +23.0% |
| OWZ | 6.70 | 8.72 | +2.02 | +30.2% |

这是最关键的结果：ML和OWZ都显示 GGW 后 MDR 热带气旋生成增加，方向一致，幅度也在同一量级。

### 5.3 路径点密度

| 方法 | CTL | GGW | GGW-CTL | 相对变化 |
|---|---:|---:|---:|---:|
| ML 全NA路径点 | 232.13 | 280.52 | +48.38 | +20.8% |
| OWZ 全NA路径点 | 422.40 | 469.07 | +46.67 | +11.0% |
| ML MDR路径点 | 101.23 | 117.17 | +15.94 | +15.7% |
| OWZ MDR路径点 | 142.73 | 175.00 | +32.27 | +22.6% |
| ML Caribbean/Gulf箱区 | 44.67 | 56.90 | +12.23 | +27.4% |
| OWZ Caribbean/Gulf箱区 | 68.87 | 84.55 | +15.69 | +22.8% |
| ML open-ocean箱区 | 105.20 | 131.10 | +25.90 | +24.6% |
| OWZ open-ocean箱区 | 148.40 | 184.28 | +35.88 | +24.2% |

路径密度图显示，ML与OWZ都捕捉到 GGW 后热带北大西洋路径点增加，尤其在 `10N-20N`、西部和中部北大西洋有增强信号。

## 6. 可以写进报告的解释

这套实验说明：虽然ML检测器没有直接使用OWZ作为输入特征，也没有用OWZ轨迹作为正标签训练，但当它被应用到CESM CTL/GGW输出后，仍然得到与OWZ一致的关键结论：GGW使北大西洋JAS热带气旋活动增强，MDR生成数增加，路径密度向热带北大西洋和加勒比/墨西哥湾方向增强。

因此，ML方法可以作为传统OWZ方法的独立验证工具。它不是简单复刻OWZ阈值，而是从IBTrACS/ERA5中学习“TC样本的大气结构特征”，再迁移到CESM模式输出上。

## 7. 需要注意的限制

1. ERA5环境场是daily资料，训练阶段的大尺度环境特征不是严格6小时资料。
2. ML检测的候选中心仍依赖宽松涡度筛选，因此不是完全端到端的格点分类。
3. 当前ML阈值 `prob_tc >= 0.55` 偏保守，导致绝对轨迹数低于OWZ；如果报告想更接近OWZ数量，可做阈值敏感性实验。
4. ML轨迹连接使用简单的800 km最近邻连接，后续可改进为更精细的多目标追踪。

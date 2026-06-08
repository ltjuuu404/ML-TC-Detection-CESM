# Model Notes

## Detection Models

The detection models classify candidate environments as TC-like or background-like.

Included files:

- `models/detection_best.joblib`: best detector selected for CESM transfer.
- `models/detection_svc_rbf.joblib`: RBF-SVC detector.
- `models/detection_svc_linear.joblib`: Linear-SVC detector.
- `models/detection_logistic.joblib`: logistic regression detector.
- `models/detection_random_forest.joblib`: random forest detector.

Best held-out performance:

- RBF-SVC F1: 0.976
- RBF-SVC ROC-AUC: 0.996

## Track-Prediction Models

The path models predict displacement components at 6, 12, 18 and 24 h.

Included models:

- `path_lr_lead*.joblib`: linear regression baselines.
- `path_svr_linear_lead*.joblib`: linear SVR models.
- `path_svr_rbf_lead*.joblib`: RBF SVR models.
- `path_svr_poly_lead*.joblib`: polynomial SVR models.
- `path_svr_sigmoid_lead*.joblib`: sigmoid SVR models.

The linear SVR models are the most robust in this experiment.

## 中文说明

检测模型用于判断候选环境是否具有热带气旋特征；路径模型用于预测未来 6/12/18/24 h 的东西向和南北向位移。`detection_best.joblib` 是 CESM 应用阶段使用的最优检测模型。


🛠️ Predictive Maintenance & Machine Failure Prediction
A Complete End-to-End ML Pipeline for Classification, Regression, Anomaly Detection & Clustering
📌 Overview

Streamlit Deploy: https://apppredectivemaintenance-5cw3xqbby8bwcjy9gkmd4k.streamlit.app/

This project implements a full predictive maintenance system using machine-learning techniques to:

✅ Predict machine failures within 30 days (classification)

✅ Predict remaining tool life / MTBF (regression)

✅ Detect machine anomalies (PyOD)

✅ Segment machine behavior into clusters (H2O K-Means)

The dataset used is the AI4I 2020 Predictive Maintenance Dataset, containing 10,000+ samples with sensor data such as temperatures, torque, rotational speed, tool wear, and product type.

A major challenge in the dataset is a 3.39% failure rate, making it highly imbalanced — this is addressed using SMOTE.


⚙️ 1. Machine Failure Prediction (Classification)

🎯 Objective:

Predict if a machine will fail within the next 30 days.


🔍 Key Insights

No missing data.

Failure rate: 3.39% → severe class imbalance.

Engineered features:

Temp_diff = Process temperature – Air temperature

Wear_per_torque = Tool wear / Torque

Failure types include: TWF, HDF, PWF, OSF, RNF.


⏳ MTBF Approximation

True MTBF timestamps do not exist in this dataset, so a proxy metric was built using:

Average tool wear at failure.

Product Type	MTBF Proxy (min of tool wear)
L	148.9 min
H	143.9 min
M	129.3 min


➡️ L-type machines tolerate more wear, while M-type fail earlier.


🤖 Modeling


Model: RandomForestClassifier

Handling imbalance: SMOTE

Evaluation: Stratified K-Fold, metric = PR-AUC


📊 Performance


Cross-Validation PR-AUC: 0.710 ± 0.054
Test PR-AUC: 0.698

Classification Report (Failure class)

Precision: 0.48

Recall: 0.79

F1-score: 0.60

High recall ensures most failures are caught → crucial for maintenance.


⚙️ 2. MTBF / Tool Wear Prediction (Regression)

🎯 Objective:


Predict how much tool wear remains before failure (remaining useful life).

Two regression pipelines were built.


🅐 PyCaret Automated ML


PyCaret compared multiple regression models.

Best model: ExtraTreesRegressor

Exported model: model_time_pycaret.pkl


🅑 Custom Scikit-Learn Pipeline


Includes:

StandardScaler for numerical features

OneHotEncoder for categorical features

RandomForestRegressor with cross-validation


📊 Performance (Test Set)

MAE: 0.996

RMSE: 2.472

R²: 0.999


⭐ Feature Importance (Top)

Wear_per_torque

Torque [Nm]

Tool wear [min]

Process temperature

Exceptional accuracy (R² = 0.999) → enables precise maintenance scheduling.


⚙️ 3. Anomaly Detection (PyOD)

Library: PyOD

Model: KNN

Data cleaned & imputed (numerical only)

Outputs:

anomaly_label (1 = anomaly)

anomaly_score


📊 Results


1000 anomalies detected (~10% of dataset)

Exported with appended anomaly results.


⚙️ 4. Machine Behavior Clustering (H2O K-Means)


Library: h2o

Clustering model: H2OKMeansEstimator, k=2


📊 Cluster Distribution


Cluster 0: 9819 machines

Cluster 1: 181 machines

Cluster 1 represents a distinct machine behavior group, likely associated with abnormal operation.

Full cluster profiles exported to:
Predictive_Maintenance_cluster_profiles.csv


🧠 Conclusion


This project delivers a comprehensive Predictive Maintenance pipeline, including:

✔ Failure prediction with strong recall (79%)

✔ MTBF estimation using engineered features

✔ Tool life regression with near-perfect R²

✔ Anomaly detection using PyOD

✔ Operational clustering using H2O

These results enable factories and industrial operations to:

Reduce downtime

Schedule maintenance intelligently

Detect abnormal machine behavior early

Monitor operational patterns

Optimize resource planning


✍️ Author:

Juan Diaz

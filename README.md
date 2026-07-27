# Predictive Maintenance for Industrial Equipment Using Time Series Sensor Data

## Project Overview
This project builds a machine learning solution to predict the Remaining Useful Life (RUL) of jet engines using time series sensor data from NASA's C-MAPSS Turbofan Engine Dataset. By forecasting how many operating cycles remain before an engine is likely to fail, the model enables proactive maintenance scheduling, reducing unplanned downtime and maintenance costs.

## Business Context
Unplanned equipment failures in industrial settings lead to costly downtime, safety risks, and inefficient maintenance operations. This project simulates a real-world predictive maintenance scenario using sensor data from jet engines, a critical application in aerospace and manufacturing.

## Dataset
NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset - 4 sub-datasets (FD001-FD004), each simulating engines under different operating conditions and fault modes. Each row represents one engine at one operating cycle, with 21 sensor readings plus operational settings.

## Technical Approach

### 1. Data Preprocessing
- Loaded and merged all 4 sub-datasets (FD001-FD004)
- Calculated RUL (Remaining Useful Life) as the target variable
- Removed constant sensors carrying no useful signal
- Handled missing values (forward/backward fill within engine groups) and capped outliers using IQR

### 2. Feature Engineering
- Engineered rolling mean, rolling standard deviation, and rate-of-change (diff) features per sensor over a 5-cycle window to capture degradation trends
- Automated feature selection (SelectKBest) confirmed engineered rolling features dominate the top predictive signals across all datasets

### 3. Modeling
- Compared three models: Random Forest, Gradient Boosting, XGBoost
- Time-ordered train/test split (no shuffling) to prevent data leakage
- 5-fold TimeSeriesSplit cross-validation for robust evaluation
- Hyperparameter tuning via GridSearchCV (n_estimators, max_depth, learning_rate)

### 4. Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 Score

### 5. Interpretability & Business Output
- Feature importance analysis per dataset
- Predicted RUL converted into actionable maintenance categories:
  - Immediate Maintenance Required (RUL <= 10)
  - Plan Maintenance (RUL <= 50)
  - Inspection Recommended (RUL <= 100)
  - Normal - Continue Monitoring (RUL > 100)

## Results

| Dataset | MAE | RMSE | R2 |
|---|---|---|---|
| FD001 | 38.37 | 51.05 | 0.579 |
| FD002 | 34.22 | 48.25 | 0.509 |
| FD003 | 51.70 | 75.09 | 0.445 |
| FD004 | 52.45 | 66.67 | 0.387 |

## Stretch Goals Implemented
- Degradation trend visualization (rolling mean vs. raw sensor values)
- Automated feature selection pipeline (SelectKBest)
- Real-time prediction simulation (streaming sensor input)

## Key Findings
- Engineered rolling-mean/std features dominate the top 10 predictive features across all datasets, validating the feature engineering approach
- FD002 and FD004 (multiple operating conditions) are harder to predict than FD001/FD003 (single operating condition)
- A negative RUL prediction observed in FD002 was corrected via clipping (RUL cannot be negative in practice)

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Joblib (model persistence)

## Setup Instructions
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Open turbofan_rul_prediction.ipynb in Jupyter Notebook, JupyterLab, or Google Colab
4. Run all cells in order

## Colab Notebook Link
[Add your Colab share link here]

## Video Demonstration
[Add your unlisted YouTube link here]

## Future Scope
- Incorporate deep learning models (LSTM/GRU) for sequence modeling
- Explore anomaly detection techniques to complement RUL prediction
- Deploy as a real-time monitoring web application
- Visualize prediction intervals for uncertainty quantification

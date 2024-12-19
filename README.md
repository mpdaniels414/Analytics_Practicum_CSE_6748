# Turbofan Degradation Analysis: Predicting Remaining Useful Life (RUL) and Fault Classification

## Overview
This project is a comprehensive analysis of turbofan engines, focusing on predicting the **Remaining Useful Life (RUL)** and identifying faulty engines based on sensor data. Utilizing NASA's CMAPSS dataset, the project integrates data exploration, feature engineering, regression, and classification models to address complex operational conditions and fault modes.

---

## Objectives
1. **Predict Remaining Useful Life (RUL)**:
   - Develop regression models to estimate the remaining operational life of turbofan engines under varying conditions.
2. **Classify Faulty Engines**:
   - Build classification models to detect engine degradation based on sensor data.

---

## Key Achievements
1. **Regression Performance**:
   - Improved RUL prediction for FD001 (single fault mode) using technical indicators, increasing **R²** from 0.5246 to 0.786 and reducing **MAE** to 14.25.
   - Applied the model to datasets with greater complexity (FD002, FD003, FD004) and identified challenges related to multicollinearity and variability.
2. **Classification Accuracy**:
   - Achieved balanced accuracy of **0.9761** with a **Naïve Bayes** classifier for detecting faulty engines.
   - Prioritized minimizing Type II errors to avoid critical misclassifications.
3. **Innovative Indicators**:
   - Integrated indicators such as RSI, CCI, Bollinger Bands Percentage, and Stochastic Oscillators to enhance predictive accuracy.
4. **Comprehensive Analysis**:
   - Detailed feature selection, correlation analysis, and validation metrics to ensure robust model performance.

---

## Getting Started

### Prerequisites
Make sure you have the following packages installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- imbalanced-learn
- xgboost
- tensorflow

Install them using the following pip command:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels imbalanced-learn xgboost tensorflow


---

## Data
The project utilizes NASA’s CMAPSS dataset:
- Four subsets: FD001, FD002, FD003, FD004, each representing different operational conditions and fault modes.
- Key features include temperature, pressure, speed, and flow sensors.

Data files and further details are available at the [NASA Prognostics Center of Excellence Data Repository](http://ti.arc.nasa.gov/projects/data_prognostics).

---

## Steps to Reproduce

### 1. Data Exploration and Feature Selection
- Run **`Data_Exploration.ipynb`** to analyze sensor data distributions, generate correlation heatmaps, and conduct VIF analysis.
- Features like `T24`, `T30`, `P30`, `phi`, and `BPR` were identified as critical for RUL predictions.

### 2. Initial RUL Prediction
- Use **`Initial_RUL_Prediction.ipynb`** to train regression models such as Ridge, Lasso, Neural Network, and XGBoost.
- Evaluate performance using metrics: **MAE**, **MSE**, and **R²**.

### 3. Enhancing Models with Indicators
- Run **`RUL_Prediction_Indicators.ipynb`** to integrate technical indicators:
  - Relative Strength Index (RSI)
  - Commodity Channel Index (CCI)
  - Bollinger Bands Percentage (BB%)
  - Stochastic Oscillators
- Observe significant performance improvements.

### 4. Fault Classification
- Use **`Classification.ipynb`** to train models including Naïve Bayes, QDA, and Random Forest.
- Adjust thresholds for balanced accuracy and minimal Type II error.

---

## Results Summary
### FD001 (Single Operating Condition)
- **R²**: 0.786  
- **MAE**: 14.25  
- **Score**: 583  

### FD002 (Multiple Conditions)
- **R²**: 0.689  
- **MAE**: 22.55  
- **Score**: 25,616  

### FD003 (Single Condition, Multiple Fault Modes)
- **R²**: 0.56  
- **MAE**: 18.18  
- **Score**: 4,677  

### FD004 (Multiple Conditions, Multiple Fault Modes)
- **R²**: 0.461  
- **MAE**: 30.9  
- **Score**: 36,608  

---

## Model Insights
1. **Regression**:
   - Neural Networks performed best for regression tasks, effectively capturing non-linear relationships.
   - Indicators like RSI and BB% enhanced predictive accuracy significantly.
2. **Classification**:
   - Naïve Bayes achieved high balanced accuracy of 0.9761 after threshold optimization.
   - Minimizing Type II errors was prioritized to prevent misclassification of faulty engines.

---

## Challenges
- **Multicollinearity**: High correlations among sensors required careful feature selection.
- **Data Variability**: FD002 and FD004 presented greater complexity due to multiple operating conditions and fault modes.
- **Noise**: Sensor data exhibited irregular distributions, complicating predictions.

---

## Team Contributions
- **Michael Daniels**: Developed regression models and RUL prediction analysis.
- **Cory Bowersox**: Conducted feature engineering and indicator logic implementation.
- **Baris Kopruluoglu**: Built classification models and optimized performance metrics.

---

## References
1. NASA Prognostics Data Repository: [Link](http://ti.arc.nasa.gov/projects/data_prognostics)
2. Saxena, K., Goebel, D., et al., “Damage Propagation Modeling for Aircraft Engines.”
3. Wang, T., Yu, J., et al., “Similarity-based Prognostics for RUL Estimation.”

---

## Conclusion
This project demonstrates the potential of advanced machine learning techniques in predictive maintenance for turbofan engines. By incorporating technical indicators and addressing challenges like multicollinearity and noise, the team achieved robust results for RUL prediction and fault classification. These insights pave the way for more reliable and efficient maintenance models, enhancing aviation safety and operational efficiency.

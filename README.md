# Analytics_Practicum_CSE_6748
My final master's project at Georgia Tech


## Overview

This project provides a comprehensive analysis and prediction pipeline for machinery data. It includes steps for conducting VIF tests, correlation models, heat mapping, feature distribution analysis, and building various prediction and classification models.

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

You can install these packages using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels imbalanced-learn xgboost tensorflow

Additional Libraries
Here are some of the specific libraries and modules you may need to import:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, accuracy_score, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


Data Files
You can find all the relevant data files in the CMAPPSDATA folder included in the project directory.

Running the Analysis
Step 1: Data Analysis
Start by running Data_Analysis.ipynb. This notebook will help you conduct VIF tests, correlation models, heat mapping, and feature distribution analysis.

Swap out the files for whichever ones you'd like to review (e.g., FD001, FD002) while following the naming convention.
Read the training, test, and RUL data:

# Swap out the files for whichever ones you'd like to review. I.e. FD001, FD002, but make sure to follow the naming convention
# Read the training data
train_file_path = 'train_FD001.txt'
train_data = pd.read_csv(train_file_path, delim_whitespace=True, header=None, names=headers)

# Read the test data
test_file_path = 'test_FD001.txt'
test_data = pd.read_csv(test_file_path, delim_whitespace=True, header=None, names=headers)

# Read the RUL data
rul_file_path = 'RUL_FD001.txt'
rul_data = pd.read_csv(rul_file_path, header=None, names=['RUL'])

Perform the analysis for all four FD files.
Step 2: Initial RUL Prediction
After completing the data analysis, run Initial_RUL_Prediction.ipynb. This notebook contains the different models created before adding indicators.

Step 3: RUL Prediction with Indicators
Next, run RUL_Prediction_Indicators.ipynb. This notebook demonstrates how indicators add value to the prediction functions.

Step 4: Classification Model
Finally, run Classification.ipynb to see the classification model developed in the project.

Conclusion
By following these steps, you will be able to analyze the data, build prediction models, classification models, and evaluate the performance of these models with and without indicators. 
We hope you enjoy this project, we had a blast putting it all together this summer for you!

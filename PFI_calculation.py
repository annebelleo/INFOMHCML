import os

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import ast
from lime import submodular_pick

file_path = 'data/result_PFI'

# Load your dataset
data = pd.read_csv('data/student-por.csv', delimiter=';')

# Replace categorical values with numerical values
data = data.replace(to_replace=['M', 'F'], value=[0, 1])  # sex
data = data.replace(to_replace=['GP', 'MS'], value=[0, 1])  # school
data = data.replace(to_replace=['A', 'T'], value=[0, 1])  # Pstatus
data = data.replace(to_replace=['GT3', 'LE3'], value=[0, 1])  # famsize
data = data.replace(to_replace=['U', 'R'], value=[0, 1])  # address
data = data.replace(to_replace=['father', 'mother'], value=[0, 1])  # guardian
data = data.replace(to_replace=['at_home', 'health', 'other', 'services', 'teacher'], value=[0, 1, 2, 3, 4])  # Mjob, Fjob
data = data.replace(to_replace=['course', 'other', 'home', 'reputation'], value=[0, 1, 2, 3])  # reason
data = data.replace(to_replace=['no', 'yes'], value=[0, 1])  # various yes/no columns

def PFI(X, labels, model, base_rmse):
  results = []

  for feature in X.columns:
      X_permuted = X.copy()
      X_permuted[feature] = np.random.permutation(X_permuted[feature])
      predictions = model.predict(X_permuted)
      new_rmse = root_mean_squared_error(labels, predictions)
      rmse_increase = new_rmse - base_rmse
      results.append((feature, rmse_increase))

  results_df = pd.DataFrame(results, columns=['Feature', 'RMSE Increase'])
  results_df = results_df.sort_values(by='RMSE Increase', ascending=False).reset_index(drop=True)

  return results_df

# Give it the trained model, the training set, the test set and the variable that we want to predict and it will return the SHAP values(at the bottom you can find an example)
def calculate_and_save_pfi_values(model, X_train, X_test, Y_TEST):

    # Extract column names from X_train
    columns = X_train.columns.tolist()
    # Extract the name of the variable to predict from X_predict
    predict_column = Y_TEST.name
    # Construct the path
    column_str = '_'.join(columns)
    full_path = f"{file_path}/MODEL_{type(model).__name__}_FEATURES_{column_str}_PRED_{predict_column}.csv"

    if not os.path.exists(full_path):

        base_predictions = model.predict(X_test)
        base_rmse = root_mean_squared_error(Y_TEST, base_predictions)
        pfi_results_df = PFI(X_test, Y_TEST, model, base_rmse)


        # Save the PFI values to a file
        pfi_results_df.to_csv(full_path, index=False)

    else:
        pfi_values = pd.read_csv(full_path)
        pfi_results_df = pfi_values.iloc[1:]
    return pfi_results_df

def plot_pfi_values(pfi_values):
    plt.figure(figsize=(10, 6))
    plt.barh(pfi_values['Feature'], pfi_values['RMSE Increase'], color='skyblue')
    plt.xlabel('RMSE Increase')
    plt.ylabel('Feature')
    plt.title('Feature Importances by PFI')
    plt.gca().invert_yaxis()
    plt.show()


# Prepare features and target variable
#features = data.drop(columns=['G3'])
features = data[['G2', 'G1']]
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regr = SVR().fit(X_train, y_train)

pfi_values = calculate_and_save_pfi_values(regr, X_train, X_test, y_test)
plot_pfi_values(pfi_values)
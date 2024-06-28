import os

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statistics

file_path = 'data/result_PFI'
pfi_loops = 100 #for more robust calculations increase this value

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
      average_error = []
      for run in range(pfi_loops):
          X_permuted = X.copy()
          X_permuted[feature] = np.random.permutation(X_permuted[feature])
          predictions = model.predict(X_permuted)
          new_rmse = root_mean_squared_error(labels, predictions)
          rmse_increase = new_rmse - base_rmse
          average_error.append(rmse_increase)
      results.append((feature, statistics.mean(average_error)))

  results_df = pd.DataFrame(results, columns=['Feature', 'RMSE Increase'])
  results_df['RMSE Increase'] = results_df['RMSE Increase'].abs()
  results_df = results_df.sort_values(by='RMSE Increase', ascending=False).reset_index(drop=True)

  return results_df

# Give it the trained model, the training set, the test set and the variable that we want to predict and it will return the SHAP values(at the bottom you can find an example)
def calculate_and_save_pfi_values(model, X_test, y_test, predict_variable):

    # Extract column names from X_train
    columns = X_test.columns.tolist()
    # Extract the name of the variable to predict from X_predict
    predict_column = predict_variable.name
    # Construct the path
    column_str = '_'.join(columns)
    full_path = f"{file_path}/MODEL_{type(model).__name__}_FEATURES_{column_str}_PRED_{predict_column}.csv"

    # Create the result_LIME directory if it doesn't exist
    os.makedirs('data/result_LIME', exist_ok=True)

    # if the path contains more than 260 chars use a shorter string
    if len(full_path) > 260:
        full_path = f"{file_path}/MODEL_{type(model).__name__}_PRED_{predict_column}"

    if not os.path.exists(full_path):

        base_predictions = model.predict(X_test)
        base_rmse = root_mean_squared_error(y_test, base_predictions)
        pfi_results_df = PFI(X_test, y_test, model, base_rmse)

        # Save the PFI values to a file
        pfi_results_df.to_csv(full_path, index=False)

    else:
        pfi_results_df = pd.read_csv(full_path)
    return pfi_results_df

def plot_pfi_values(pfi_values):
    plt.figure(figsize=(10, 6))
    plt.barh(pfi_values['Feature'], pfi_values['RMSE Increase'], color='skyblue')
    plt.xlabel('RMSE Increase')
    plt.ylabel('Feature')
    plt.title('Feature Importances by PFI')
    plt.gca().invert_yaxis()
    plt.show()

'''
# Prepare features and target variable
#features = data.drop(columns=['G3'])
features = data
X = features.drop(columns=['G1', 'G2', 'G3', 'sex'])
y = data['sex']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#use SVR for the poster presentation (SVC has a different structure for shap)
regr = SVR().fit(X_train, y_train)
pfi_values = calculate_and_save_pfi_values(regr, X_train, X_test, y)
plot_pfi_values(pfi_values)
'''
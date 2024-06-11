import shap
import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np


file_path = 'data/result_SHAP/shap_values.csv'

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

# Prepare features and target variable
features = data.drop(columns=['G3'])
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regr = SVR().fit(X_train, y_train)

# Make predictions
pred = regr.predict(X_test)
print(metrics.mean_squared_error(y_test, pred))  # Calculate and print MSE

if not os.path.exists(file_path):
    # Initialize the SHAP explainer
    explainer = shap.KernelExplainer(regr.predict, X_train)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    # Create the result_SHAP directory if it doesn't exist
    os.makedirs('data/result_SHAP', exist_ok=True)

    # Save the SHAP values to a file
    shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_values_df.to_csv(file_path, index=False)

else:
    shap_values_df = pd.read_csv(file_path)
    shap_values = shap_values_df.values

# Summary plot
shap.summary_plot(shap_values, X_test)

# Bar plot
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X_test.columns)

# Dependence plot for a specific feature
shap.dependence_plot('Dalc', shap_values, X_test)  # Change 'Dalc' to any feature of your interest







# Get the absolute SHAP values
shap_values_abs = np.abs(shap_values)

# Calculate the mean of absolute SHAP values for each feature
mean_shap_values = shap_values_abs.mean(axis=0)

# Sort the features based on their mean absolute SHAP values and select the top 10
top_features = sorted(range(len(mean_shap_values)), key=lambda i: mean_shap_values[i], reverse=True)[:10]

# Get the feature names of the top 10 features
top_feature_names = X_test.columns[top_features].tolist()

# Include 'sex' feature if it's not already in the top 10 features
if 'sex' not in top_feature_names:
    top_feature_names.append('sex')

# Filter X_test to include only the selected features
X_test_selected = X_test[top_feature_names]

# Filter SHAP values to include only the selected features
shap_values_selected = shap_values[:, X_test.columns.isin(top_feature_names)]

# Plot the summary plot with the selected features
shap.summary_plot(shap_values_selected, X_test_selected, plot_type="bar", feature_names=top_feature_names)

import shap
import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
features = data.drop(columns=['G3', 'sex'])
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
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Dependence plot for a specific feature
shap.dependence_plot('Dalc', shap_values, X_test)  # Change 'Dalc' to any feature of your interest
import shap
import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np


file_path = 'data/result_SHAP'

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

# Give it the trained model, the training set, the test set and the variable that we want to predict and it will return the SHAP values(at the bottom you can find an example)
#recalculate = True if you want to recalculate SHAP values (for example if you have already calculated them with a model, but you change the hyperparameters afterwards so you need to recalculate the values)
def calculate_and_save_shap_values(model, X_train, X_test, X_predict, recalculate = False):

    # Extract column names from X_train
    columns = X_train.columns.tolist()
    # Extract the name of the variable to predict from X_predict
    predict_column = X_predict.name
    # Construct the path
    column_str = '_'.join(columns)
    full_path = f"{file_path}/MODEL_{type(model).__name__}_FEATURES_{column_str}_PRED_{predict_column}.csv"

    if os.path.exists(full_path) and not recalculate:

        shap_values_df = pd.read_csv(full_path)
        shap_values = shap_values_df.values

    else:
        # Initialize the SHAP explainer
        explainer = shap.KernelExplainer(model.predict, X_train)

        # Calculate SHAP values for the test set
        shap_values = explainer.shap_values(X_test)

        # Create the result_SHAP directory if it doesn't exist
        os.makedirs('data/result_SHAP', exist_ok=True)

        # Save the SHAP values to a file
        shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
        shap_values_df.to_csv(full_path, index=False)
    return shap_values

#plot things with SHAP
def plot_shap_values(shap_values, X_test):
    # Summary plot
    shap.summary_plot(shap_values, X_test)

    # Bar plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X_test.columns)

    # Dependence plot for a specific feature
    #shap.dependence_plot('Dalc', shap_values, X_test)  # Change 'Dalc' to any feature of your interest

#select the most important feature, will be useful when we want to retrain the model without the most important proxies
def most_important_feature(shap_values, X_test, num_features):
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Create a DataFrame with feature names and their mean absolute SHAP values
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'MeanAbsShapValue': mean_abs_shap_values
    })

    # Sort the DataFrame by mean absolute SHAP value in descending order
    feature_importance = feature_importance.sort_values(by='MeanAbsShapValue', ascending=False)

    # Select the top 5 most important features
    top_5_features = feature_importance.head(num_features)

    return top_5_features

# Prepare features and target variable
#features = data.drop(columns=['G3'])
features = data[['G2', 'G1']]
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regr = SVR().fit(X_train, y_train)

shap_values = calculate_and_save_shap_values(regr, X_train, X_test, y)
#if you want to calculate again the SHAP values
#shap_values = calculate_and_save_shap_values(regr, X_train, X_test, y, True)
plot_shap_values(shap_values, X_test)
#1 is the number of top features that you want to return (the most impacting ones on the final outcome)
print(most_important_feature(shap_values, X_test, 1))
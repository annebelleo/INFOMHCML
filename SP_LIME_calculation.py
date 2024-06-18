import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
from collections import Counter
import ast


file_path = 'data/result_LIME'

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
def calculate_and_save_lime_values(model, X_train, X_test, X_predict):

    # Extract column names from X_train
    columns = X_train.columns.tolist()
    # Extract the name of the variable to predict from X_predict
    predict_column = X_predict.name
    # Construct the path
    column_str = '_'.join(columns)
    full_path = f"{file_path}/MODEL_{type(model).__name__}_FEATURES_{column_str}_PRED_{predict_column}.csv"

    if not os.path.exists(full_path):
        # Initialize the SHAP explainer
        explainer = LimeTabularExplainer(X_train.values, mode='regression', feature_names=X_train.columns,
                                         training_labels=y_train, discretize_continuous=True)
        lime_values = []
        for i in range(X_test.shape[0]):
            # Generate explanation for the current test instance
            exp = explainer.explain_instance(X_test.iloc[i].values, model.predict)
            lime_values.append(exp.as_list())

        lime_values_formatted = []
        for value in lime_values:
            parsed_features = []
            for feature in value:
                parts = feature[0].split()
                for col in X.columns:
                    if col in parts[0]:
                        feature_name = col
                feature_value = feature[1]
                parsed_features.append((feature_name, feature_value))
            lime_values_formatted.append(parsed_features)

        lime_df = pd.DataFrame(lime_values_formatted)
        lime_df.to_csv(full_path, index=False)

    else:
        lime_values = pd.read_csv(full_path)
        lime_values = lime_values.iloc[1:]
    return lime_values

#plot things with SHAP
def plot_lime_values(lime_df, X_test):
    # Here we simply print the first few LIME explanations as an example
    print('test')
def parse_feature(feature_str):
    # This function parses the string representation of the tuple
    try:
        return ast.literal_eval(feature_str)
    except (ValueError, SyntaxError):
        return None


def most_important_feature(lime_values, num_features=1):
    most_important_features = []

    for instance in lime_values:
        for feature in instance:
            feature[1]


        parsed_features = [parse_feature(f) for f in instance]
        # Filter out any None values that failed to parse
        parsed_features = [f for f in parsed_features if f is not None]

        # Sort the features based on their absolute importance
        sorted_features = sorted(parsed_features, key=lambda x: abs(x[1]), reverse=True)
        if sorted_features:
            most_important_features.append(sorted_features[0][0])

    # Count the frequency of each feature
    feature_counter = Counter(most_important_features)
    return feature_counter.most_common(num_features)


# Prepare features and target variable
#features = data.drop(columns=['G3'])
features = data[['G2', 'G1']]
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regr = SVR().fit(X_train, y_train)

lime_values = calculate_and_save_lime_values(regr, X_train, X_test, y)
plot_lime_values(lime_values, X_test)
print(most_important_feature(lime_values, 1))
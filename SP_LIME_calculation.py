import os

import pandas as pd
from lime import submodular_pick
from lime.lime_tabular import LimeTabularExplainer

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
def calculate_and_save_lime_values(model,X, X_train, X_test, predict_variable):

    # Extract column names from X_train
    columns = X_train.columns.tolist()
    # Extract the name of the variable to predict from X_predict
    predict_column = predict_variable.name
    # Construct the path
    column_str = '_'.join(columns)
    full_path = f"{file_path}/MODEL_{type(model).__name__}_FEATURES_{column_str}_PRED_{predict_column}"

    # Create the result_LIME directory if it doesn't exist
    os.makedirs('data/result_LIME', exist_ok=True)

    # if the path contains more than 260 chars use a shorter string
    if len(full_path) > 260:
        full_path = f"{file_path}/MODEL_{type(model).__name__}_PRED_{predict_column}"

    if not os.path.exists(full_path + f'.csv'):

        explainer = LimeTabularExplainer(X_train.values, mode='regression', feature_names=X_train.columns,
                                         training_labels=X_test, discretize_continuous=True)

        exps = submodular_pick.SubmodularPick(explainer, X_train.to_numpy(), model.predict, method='full', num_features=X_train.shape[1],
                                       num_exps_desired=4)

        lime_values = []
        for exp in exps.sp_explanations:
            lime_values.append(exp.as_list())

        lime_values_formatted = []
        for value in lime_values:
            parsed_features = []
            for feature in value:
                parts = feature[0].split()
                for col in X.columns:
                    if col in parts:
                        feature_name = col
                feature_value = feature[1]
                parsed_features.append((feature_name, feature_value))
            lime_values_formatted.append(parsed_features)

        #checking that limes values are listed by importance
        lime_values_feature_ranking = []
        for element in lime_values_formatted:
            sorted_values = sorted(element, key=lambda x: abs(x[1]), reverse=True)
            lime_values_feature_ranking.append([item[0] for item in sorted_values])


        # Assuming sp_obj.sp_explanations is a list of explanations
        figures = [exp.as_pyplot_figure() for exp in exps.sp_explanations]

        # save each figure individually
        for i, fig in enumerate(figures):
            fig.savefig(full_path + f'exp_{i + 1}_figure_{i + 1}.png')
            fig.show()  # or plt.show(fig) depending on the exact return type
            #fig.show()  # or plt.show(fig) depending on the exact return type
            #[exp.as_pyplot_figure() for exp in exps.sp_explanations]

        #lime_values_formatted = []
        lime_df = pd.DataFrame(lime_values_feature_ranking)
        lime_df.to_csv(full_path + f'.csv', index=False)

    else:
        lime_df = pd.read_csv(full_path+ f'.csv')
    return lime_df

#retrieve feature list importance from the first explanation
def retrieve_feature_list(lime_values):
    return [item for item in lime_values.iloc[0]]

'''
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
print(retrieve_feature_list(lime_values))
'''
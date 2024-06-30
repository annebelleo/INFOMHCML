import matplotlib.pyplot as plt
import pandas as pd
import shap
from lime import submodular_pick
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error
import numpy as np
import statistics


pfi_loops = 100 #for more robust calculations increase this value
# set this variable to the number of proxies for the protected attribute
number_of_most_important_feature = 5
# features not used for calculating the final grade or proxies
features_not_used = ['G1', 'G2', 'G3']
# Categorical features
features = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
            'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
            'schoolsup', 'famsup', 'fatherd', 'activities', 'nursery', 'higher', 'internet',
            'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
numeric_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'fatherd', 'activities', 'nursery', 'higher', 'internet', 'romantic']

one_hot_encoded_features = {
    'Mjob': ['Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher'],
    'Fjob': ['Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher'],
    'reason': ['reason_home', 'reason_other', 'reason_reputation'],
    'guardian': ['guardian_mother', 'guardian_other']
}

categorical_features_preprocessed = {
    'Mjob': ['Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher'],
    'Fjob': ['Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher'],
    'reason': ['reason_home', 'reason_other', 'reason_reputation'],
    'guardian': ['guardian_mother', 'guardian_other'],
    'school': ['school_MS'],
    'sex': ['sex_M'],
    'address': ['address_U'],
    'famsize': ['famsize_LE3'],
    'Pstatus': ['Pstatus_T'],
    'schoolsup': ['schoolsup_yes'],
    'famsup': ['famsup_yes'],
    'fatherd': ['fatherd_yes'],
    'activities': ['activities_yes'],
    'nursery': ['nursery_yes'],
    'higher': ['higher_yes'],
    'internet': ['internet_yes'],
    'romantic': ['romantic_yes']
}

categorical_features_preprocessed_pfi_shuffle = {
    'Mjob_health': ['Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher'],
    'Fjob_health': ['Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher'],
    'reason_home': ['reason_home', 'reason_other', 'reason_reputation'],
    'guardian_mother': ['guardian_mother', 'guardian_other'],
}


pfi_all_categorical_feature = ['Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_other', 'reason_reputation', 'guardian_other']

def PFI(X, labels, model, base_rmse):
  results = []

  for feature in X.columns:

      if feature not in pfi_all_categorical_feature:
          average_error = []
          for run in range(pfi_loops):
              X_permuted = X.copy()
              if feature in categorical_features_preprocessed_pfi_shuffle.keys():
                  X_permuted[categorical_features_preprocessed_pfi_shuffle[feature]] = np.random.permutation(X_permuted[categorical_features_preprocessed_pfi_shuffle[feature]])
              else:
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

def plot_pfi_values(pfi_values):
    plt.figure(figsize=(10, 6))
    plt.barh(pfi_values['Feature'], pfi_values['RMSE Increase'], color='skyblue')
    plt.xlabel('RMSE Increase')
    plt.ylabel('Feature')
    plt.title('Feature Importances by PFI')
    plt.gca().invert_yaxis()
    plt.show()

def fairness_metrics(model_results):
    # Pearson correlations between sex and actual outcomes
    PCC_sex_actual = pearsonr(model_results['sex_M'], model_results['G3'])

    # Pearson correlations between sex and predicted outcomes
    PCC_sex_pred = pearsonr(model_results['sex_M'], model_results['prediction'])

    print(f'PCC (sex & actual grades):\n{(PCC_sex_actual)}')
    print('\n')
    print(f'PCC (sex & predicted grades):\n{(PCC_sex_pred)}')

    # Subset of model results per sex
    model_results_f = model_results[model_results['sex_M'] == 0]  # females (one hot encoding)
    model_results_m = model_results[model_results['sex_M'] == 1]  # males

    # Mean squared error
    MAE_f = metrics.mean_absolute_error(model_results_f['G3'], model_results_f['prediction'])
    MAE_m = metrics.mean_absolute_error(model_results_m['G3'], model_results_m['prediction'])

    # Print MAE difference per sex
    MAE_difference = (MAE_f - MAE_m)
    print('\n')
    print(f'MAE females: {round(MAE_f, 4)}')
    print(f'MAE males: {round(MAE_m, 4)}')
    print(f'MAE difference: {round(MAE_difference, 4)}')
    return ''

def change_values(series, string):
    if string in categorical_features_preprocessed.keys():
        series = series.mode().iloc[0]
    elif string in numeric_features:
        series = int(series.mean())
    else:
        raise Exception("Feature not included in categorical list or continuous list")

    return series

def proxy_removal_metrics_check(common_proxies, data, regr, X_test, y_test):
    proxies = []
    for proxy in common_proxies:
        # change the values of proxies using the same method as the paper
        if proxy not in categorical_features_preprocessed_pfi_shuffle.keys():
            X_test[proxy] = change_values(X_test[proxy], proxy)
        else:
            for key in categorical_features_preprocessed_pfi_shuffle.keys():
                X_test[key] = X_test[key].mode().iloc[0]
        # Train the model
        pred = regr.predict(X_test)
        proxies.append(proxy)

        print('\n SVC model - predicting G3 without using protected attribute sex and ' + str(proxies))
        print(metrics.mean_squared_error(y_test, pred))
        model_results = pd.DataFrame(
            {'sex_M': data.loc[X_test.index, 'sex_M'], 'G3': y_test, 'prediction': pred})

        fairness_metrics(model_results)

def finetuned_model():
    # Finetuned SVR Model
    svr_final_model = SVR(C=18, gamma='scale', epsilon=0.5).fit(X_train, y_train)
    return svr_final_model

def load_dataset():
    # Load your dataset
    data = pd.read_csv('data/student-por.csv', delimiter=';')

    # Replace categorical values with numerical values
    data = data.replace(to_replace=['M', 'F'], value=[0, 1])  # sex
    data = data.replace(to_replace=['GP', 'MS'], value=[0, 1])  # school
    data = data.replace(to_replace=['A', 'T'], value=[0, 1])  # Pstatus
    data = data.replace(to_replace=['GT3', 'LE3'], value=[0, 1])  # famsize
    data = data.replace(to_replace=['U', 'R'], value=[0, 1])  # address
    data = data.replace(to_replace=['father', 'mother'], value=[0, 1])  # guardian
    data = data.replace(to_replace=['at_home', 'health', 'other', 'services', 'teacher'],
                        value=[0, 1, 2, 3, 4])  # Mjob, Fjob
    data = data.replace(to_replace=['course', 'other', 'home', 'reputation'], value=[0, 1, 2, 3])  # reason
    data = data.replace(to_replace=['no', 'yes'], value=[0, 1])  # various yes/no columns
    return data

def load_preprocessed_dataset():

    # Load your dataset
    return pd.read_csv('data/preprocessed_student_por.csv', delimiter=',')


data = load_preprocessed_dataset()
data_unprocessed = load_dataset()
'''--------Predicting G3 using all the features except G1 and G2--------'''

# Predicting G3 using all the features except G1 and G2
features = data.drop(columns=['G3'])
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regr_final_grade_all_features = finetuned_model().fit(X_train, y_train)
pred = regr_final_grade_all_features.predict(X_test)


# Initialize the SHAP explainer SVC
explainer = shap.KernelExplainer(regr_final_grade_all_features.predict, X_test)
# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Function to sum values for each categorical feature
new_shap_values = shap_values
list_features = X_test.columns.tolist()
indices = []
for key, columns in one_hot_encoded_features.items():
    column_indices = [X_test.columns.get_loc(name) for name in columns]
    values = shap_values[:, column_indices].sum(axis=1)
    # Drop the one-hot encoded columns
    new_shap_values[:, column_indices[0]] = values
    indices.append(column_indices[1:])

integers = [num for sublist in indices for num in sublist if isinstance(num, int)]
new_shap_values = np.delete(new_shap_values, integers, axis=1)
list_features = [list_features[i] for i in range(len(list_features)) if i not in integers]


# Bar plot
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X_test.columns, max_display=20,
                       show=False)
plt.show()

# Assuming shap_values is your SHAP values matrix
feature_importance = np.mean(np.abs(new_shap_values), axis=0)
print(feature_importance)

shap_df_G3 = pd.DataFrame({
    'Name': list_features,
    'Shap_Value': feature_importance
})

shap_df_G3 = shap_df_G3.sort_values(by='Shap_Value', ascending=False)

# Print the DataFrame (optional)
print(shap_df_G3)

# Plotting using Matplotlib
plt.figure(figsize=(10, 6))
plt.barh(shap_df_G3['Name'], shap_df_G3['Shap_Value'], color='skyblue')
plt.xlabel('SHAP value')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.grid(True)
plt.show()














'''PREDICTING SEX USING SVC'''

# Predicting G3 using all the features except G1 and G2
features = data.drop(columns=['G3', 'sex_M'])
X = features
y = data['sex_M']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
class_sex_all_features = SVC(probability=True).fit(X_train, y_train)
pred = class_sex_all_features.predict(X_test)
print(metrics.accuracy_score(y_test, pred))

# Initialize the SHAP explainer SVC
explainer = shap.KernelExplainer(class_sex_all_features.predict_proba, X_test)
# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[:,:,0], X_test, plot_type="bar")
plt.show()

explainer = LimeTabularExplainer(X_test.values,
                                 mode='classification',
                                 feature_names=X_test.columns,
                                 categorical_features=[data.columns.get_loc(c) for c in
                                                       sum(one_hot_encoded_features.values(), [])],
                                 categorical_names=one_hot_encoded_features)

exps = submodular_pick.SubmodularPick(explainer, X_test.values, class_sex_all_features.predict_proba, method='full',
                                      num_features=X_test.shape[1],
                                      num_exps_desired=4)

# Explain a single instance (e.g., the first instance in X_test)
exp = explainer.explain_instance(X_test.iloc[0].values, class_sex_all_features.predict_proba, num_features=X_test.shape[1])
figure = exp.as_pyplot_figure()
figure.show()
figure.savefig('testststststs')

print('test')



#PFI

base_predictions = class_sex_all_features.predict(X_test)
base_rmse = root_mean_squared_error(y_test, base_predictions)
pfi_results_df = PFI(X_test, y_test, class_sex_all_features, base_rmse)
plot_pfi_values(pfi_results_df)



# Predicting G3 using all the features except G1 and G2
features = data.drop(columns=['G3'])
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regr_final_grade_all_features = finetuned_model().fit(X_train, y_train)
pred = regr_final_grade_all_features.predict(X_test)

'''--------Retraining the model without using sex and proxies SHAP--------'''
print('\n\n--------Retraining the model without using sex and proxies SHAP--------\n\n')

common_proxies_shap = ['Walc']
# accuracy and fairness (add fairness)
print('\n SVR model - SHAP predicting G3 without using protected attribute sex and ' + str(common_proxies_shap))
proxy_removal_metrics_check(common_proxies_shap, data, regr_final_grade_all_features, X_test, y_test)




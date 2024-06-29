import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import PFI_calculation as pfc
import SHAP_calculation as shc
import SP_LIME_calculation as spc
from scipy.stats import pearsonr

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
    if string in categorical_features and features:
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
        X_test[proxy] = change_values(X_test[proxy], proxy)
        # Train the model
        pred = regr.predict(X_test)
        proxies.append(proxy)

        print('\n SVR model - predicting G3 without using protected attribute sex and ' + str(proxies))
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
unprocessed_data = load_dataset()

'''--------Predicting G3 using all the features except G1 and G2--------'''

# Predicting G3 using all the features except G1 and G2
features = data.drop(columns=features_not_used)
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.head(60), y.head(60), test_size=0.2, random_state=42)

# Train the model
regr_final_grade_all_features = finetuned_model().fit(X_train, y_train)
pred = regr_final_grade_all_features.predict(X_test)

# accuracy and fairness (add fairness)
print('\n SVR model - predicting G3')
print(metrics.mean_squared_error(y_test, pred))

# find the most important features for predicting G3
shap_values_final_grade = shc.calculate_and_save_shap_values(regr_final_grade_all_features, X_train, X_test, y)
shap_most_important_feature_final_grade_all_feature = shc.most_important_feature(shap_values_final_grade, X_test, 5)

# find the most correlated values with sex
corr_data = data.select_dtypes(include='number')
corr = corr_data.corr()
target_feature = 'sex_M'
correlation_with_target = corr[target_feature].abs()
correlation_with_target = correlation_with_target.sort_values(ascending=False)
# Get the most correlated features
top_correlated_features = correlation_with_target.drop(['G3'] + ['sex_M']).head(10).index.tolist()

# get the intersection between SHAP and correlation
set1_grades = set(shap_most_important_feature_final_grade_all_feature['Feature'])
set2_grades = set(top_correlated_features)

common_predictor = set1_grades.intersection(set2_grades)

'''--------Predicting sex using SHAP, SP-LIME AND PFI--------'''

# Predicting SEX using all the features except G1 and G2 and G3, for lime andPFI we need to unprocessed data
features = data.drop(columns=features_not_used + ['sex_M'])
features_unprocessed = unprocessed_data.drop(columns=features_not_used + ['sex'])
X = features
X_unprocessed = features_unprocessed
y = data['sex_M']  # Predict final grade
y_unprocessed = unprocessed_data['sex']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.head(60), y.head(60), test_size=0.2, random_state=42)
# Train-test split for LIME and PFI
X_train_unprocessed, X_test_unprocessed, y_train_unprocessed, y_test_unprocessed = train_test_split(X_unprocessed.head(60), y_unprocessed.head(60), test_size=0.2, random_state=42)

# Train the model
regr_sex_all_feature = finetuned_model().fit(X_train, y_train)
pred = regr_sex_all_feature.predict(X_test)

# accuracy and fairness (add fairness)
print('\n SVR model - predicting sex')
print(metrics.mean_squared_error(y_test, pred))

# find the most important features for sex, SHAP
shap_values_sex = shc.calculate_and_save_shap_values(regr_sex_all_feature, X_train, X_test, y)
shap_most_important_feature_sex = shc.most_important_feature(shap_values_sex, X_test, number_of_most_important_feature)

# find the most important features for sex, SP_LIME
lime_values = spc.calculate_and_save_lime_values(regr_sex_all_feature, X_unprocessed, X_train_unprocessed, X_test_unprocessed, y_unprocessed)

# find the most important features for sex, PFI
pfi_values = pfc.calculate_and_save_pfi_values(regr_sex_all_feature, X_test_unprocessed, y_test_unprocessed, y_unprocessed)
pfc.plot_pfi_values(pfi_values)

# find proxies in common between these three methods
# Convert lists to sets
set1 = set(shap_most_important_feature_sex['Feature'])
set2 = set(lime_values.iloc[0][:number_of_most_important_feature])
set3 = set(pfi_values['Feature'][:number_of_most_important_feature])

# Find common elements using intersection
common_proxies = set1.intersection(set2).intersection(set3)

# Convert the result back to a list (if needed)
common_proxies = list(common_proxies)

'''--------Retraining the model without using sex and proxies SHAP--------'''
print('\n\n--------Retraining the model without using sex and proxies SHAP--------\n\n')

set1 = set(shap_most_important_feature_sex['Feature'])
set2_grades = set(shap_most_important_feature_final_grade_all_feature['Feature'])

# Find common elements using intersection
common_proxies_shap = list(set1.intersection(set2_grades))

# Predicting G3 using all the features except G1 and G2 sex and the proxies
features = data.drop(columns=features_not_used)
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.head(60), y.head(60), test_size=0.2, random_state=42)
regr = finetuned_model().fit(X_train, y_train)

# accuracy and fairness (add fairness)
print('\n SVR model - SHAP predicting G3 without using protected attribute sex and ' + str(common_proxies_shap))
proxy_removal_metrics_check(common_proxies_shap, data, regr, X_test, y_test)




'''--------Retraining the model without using sex and proxies LIME--------'''
print('\n\n--------Retraining the model without using sex and proxies LIME--------\n\n')
set1 = set(lime_values.iloc[0][:number_of_most_important_feature])
set2_grades = set(shap_most_important_feature_final_grade_all_feature['Feature'])

# Find common elements using intersection
common_proxies_lime = list(set1.intersection(set2_grades))

# Predicting G3 using all the features except G1 and G2 sex and the proxies
features = data.drop(columns=features_not_used)
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.head(60), y.head(60), test_size=0.2, random_state=42)
regr = finetuned_model().fit(X_train, y_train)

# accuracy and fairness (add fairness)
print('\n SVR model - LIME predicting G3 without using protected attribute sex and ' + str(common_proxies_lime))
# accuracy and fairness (add fairness)
proxy_removal_metrics_check(common_proxies_lime, data, regr, X_test, y_test)

'''--------Retraining the model without using sex and proxies PFI--------'''
print('\n\n--------Retraining the model without using sex and proxies PFI--------\n\n')
set1 = set(pfi_values['Feature'][:number_of_most_important_feature])
set2_grades = set(shap_most_important_feature_final_grade_all_feature['Feature'])

# Find common elements using intersection
common_proxies_pfi = list(set1.intersection(set2_grades))

# Predicting G3 using all the features except G1 and G2 sex and the proxies
features = data.drop(columns=features_not_used)
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.head(60), y.head(60), test_size=0.2, random_state=42)
regr = finetuned_model().fit(X_train, y_train)

# accuracy and fairness (add fairness)
print('\n SVR model - PFI predicting G3 without using protected attribute sex and ' + str(common_proxies_pfi))
# accuracy and fairness (add fairness)
proxy_removal_metrics_check(common_proxies_pfi, data, regr, X_test, y_test)

'''--------Retraining the model without using sex and proxies Correlation--------'''
print('\n\n--------Retraining the model without using sex and proxies Correlation--------\n\n')
set1 = set(top_correlated_features)
set2_grades = set(shap_most_important_feature_final_grade_all_feature['Feature'])

# Find common elements using intersection
common_proxies_correlation = list(set1.intersection(set2_grades))

# Predicting G3 using all the features except G1 and G2 sex and the proxies
features = data.drop(columns=features_not_used)
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.head(60), y.head(60), test_size=0.2, random_state=42)
regr = finetuned_model().fit(X_train, y_train)

# accuracy and fairness (add fairness)
print('\n SVR model - Correlation predicting G3 without using protected attribute sex and ' + str(common_proxies_correlation))
# accuracy and fairness (add fairness)
proxy_removal_metrics_check(common_proxies_correlation, data, regr, X_test, y_test)


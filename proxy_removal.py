import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

import PFI_calculation as pfc
import SHAP_calculation as shc
import SP_LIME_calculation as spc

# set this variable to the number of proxies for the protected attribute
number_of_most_important_feature = 5
# features not used for calculating the final grade or proxies
features_not_used = ['G1', 'G2', 'G3']
# Categorical features
categorical_features = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
    'guardian', 'schoolsup', 'famsup', 'activities', 'nursery', 'higher',
    'internet', 'romantic', 'Medu', 'Fedu', 'traveltime', 'studytime', 'famrel',
    'freetime', 'goout', 'Dalc', 'Walc', 'health'
]

# Continuous features
continuous_features = [
    'age', 'absences', 'G1', 'G2', 'G3', 'failures'
]


def change_values(series, string):
    if string in categorical_features:
        series = series.mode().iloc[0]
    elif string in continuous_features:
        series = int(series.mean())
    else:
        raise Exception("Feature not included in categorical list or continuous list")

    return series


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


data = load_dataset()

'''--------Predicting G3 using all the features except G1 and G2--------'''

# Predicting G3 using all the features except G1 and G2
features = data.drop(columns=features_not_used)
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regr_final_grade_all_features = SVR().fit(X_train, y_train)
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
target_feature = 'sex'
correlation_with_target = corr[target_feature].abs()
correlation_with_target = correlation_with_target.sort_values(ascending=False)
# Get the most correlated features
top_correlated_features = correlation_with_target.drop(['G3'] + ['sex']).head(10).index.tolist()

# get the intersection between SHAP and correlation
set1_grades = set(shap_most_important_feature_final_grade_all_feature['Feature'])
set2_grades = set(top_correlated_features)

common_predictor = set1_grades.intersection(set2_grades)

'''--------Predicting sex using SHAP, SP-LIME AND PFI--------'''

# Predicting SEX using all the features except G1 and G2 and G3
features = data.drop(columns=features_not_used + ['sex'])
X = features
y = data['sex']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regr_sex_all_feature = SVR().fit(X_train, y_train)
pred = regr_sex_all_feature.predict(X_test)

# accuracy and fairness (add fairness)
print('\n SVR model - predicting sex')
print(metrics.mean_squared_error(y_test, pred))

# find the most important features for sex, SHAP
shap_values_sex = shc.calculate_and_save_shap_values(regr_sex_all_feature, X_train, X_test, y)
shap_most_important_feature_sex = shc.most_important_feature(shap_values_sex, X_test, number_of_most_important_feature)

# find the most important features for sex, SP_LIME
lime_values = spc.calculate_and_save_lime_values(regr_sex_all_feature, X, X_train, X_test, y)

# find the most important features for sex, PFI
pfi_values = pfc.calculate_and_save_pfi_values(regr_sex_all_feature, X_test, y_test, y)
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

proxies = []
for proxy in common_proxies_shap:
    # change the values of proxies using the same method as the paper
    X[proxy] = change_values(X[proxy], proxy)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    regr = SVR().fit(X_train, y_train)
    pred = regr.predict(X_test)
    proxies.append(proxy)

    # accuracy and fairness (add fairness)
    print('\n SVR model - SHAP predicting G3 without using protected attribute sex and ' + str(proxies))
    print(metrics.mean_squared_error(y_test, pred))

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

proxies = []
for proxy in common_proxies_lime:
    # change the values of proxies using the same method as the paper
    X[proxy] = change_values(X[proxy], proxy)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    regr = SVR().fit(X_train, y_train)
    pred = regr.predict(X_test)
    proxies.append(proxy)

    # accuracy and fairness (add fairness)
    print('\n SVR model - LIME predicting G3 without using protected attribute sex and ' + str(proxies))
    print(metrics.mean_squared_error(y_test, pred))

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

proxies = []
for proxy in common_proxies_pfi:
    # change the values of proxies using the same method as the paper
    X[proxy] = change_values(X[proxy], proxy)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    regr = SVR().fit(X_train, y_train)
    pred = regr.predict(X_test)
    proxies.append(proxy)

    # accuracy and fairness (add fairness)
    print('\n SVR model - PFI predicting G3 without using protected attribute sex and ' + str(proxies))
    print(metrics.mean_squared_error(y_test, pred))

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

proxies = []
for proxy in common_proxies_correlation:
    # change the values of proxies using the same method as the paper
    X[proxy] = change_values(X[proxy], proxy)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    regr = SVR().fit(X_train, y_train)
    pred = regr.predict(X_test)
    proxies.append(proxy)

    # accuracy and fairness (add fairness)
    print('\n SVR model - Correlation predicting G3 without using protected attribute sex and ' + str(proxies))
    print(metrics.mean_squared_error(y_test, pred))

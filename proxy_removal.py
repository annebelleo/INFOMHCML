import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

import PFI_calculation as pfc
import SHAP_calculation as shc
import SP_LIME_calculation as spc

#set this variable to the number of proxies for the protected attribute
number_of_most_important_feature = 5
#features not used for calculating the final grade or proxies
features_not_used = ['G1', 'G2', 'G3']

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

'''--------Predicting G3 using all the features except G1 and G2--------'''

# Predicting G3 using all the features except G1 and G2
features = data.drop(columns=features_not_used)
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.head(60), y.head(60), test_size=0.2, random_state=42)

# Train the model
regr_final_grade_all_features = SVR().fit(X_train, y_train)
pred = regr_final_grade_all_features.predict(X_test)

#accuracy and fairness (add fairness)
print('\n SVR model - predicting G3')
print(metrics.mean_squared_error(y_test, pred))

#find the most important features for predicting G3
shap_values_final_grade = shc.calculate_and_save_shap_values(regr_final_grade_all_features, X_train, X_test, y)
shap_most_important_feature_final_grade_all_feature = shc.most_important_feature(shap_values_final_grade, X_test, 5)


#find the most correlated values with G3
corr_data = data.select_dtypes(include='number')
corr = corr_data.corr()
target_feature = 'G3'
correlation_with_target = corr[target_feature].abs()
correlation_with_target = correlation_with_target.sort_values(ascending=False)
# Get the top 5 most correlated features
top_correlated_features = correlation_with_target.drop(features_not_used).head(10).index.tolist()

#get the intersection between SHAP and correlation
set1_grades = set(shap_most_important_feature_final_grade_all_feature['Feature'])
set2_grades = set(top_correlated_features)

common_predictor = set1_grades.intersection(set2_grades)

'''--------Predicting sex using SHAP, SP-LIME AND PFI--------'''

# Predicting SEX using all the features except G1 and G2 and G3
features = data.drop(columns=features_not_used + ['sex'])
X = features
y = data['sex']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.head(60), y.head(60), test_size=0.2, random_state=42)

# Train the model
regr_sex_all_feature = SVR().fit(X_train, y_train)
pred = regr_sex_all_feature.predict(X_test)

#accuracy and fairness (add fairness)
print('\n SVR model - predicting sex')
print(metrics.mean_squared_error(y_test, pred))

#find the most important features for sex, SHAP
shap_values_sex = shc.calculate_and_save_shap_values(regr_sex_all_feature, X_train, X_test, y)
shap_most_important_feature_sex = shc.most_important_feature(shap_values_sex, X_test, number_of_most_important_feature)

#find the most important features for sex, SP_LIME
lime_values = spc.calculate_and_save_lime_values(regr_sex_all_feature,X, X_train, X_test, y)

#find the most important features for sex, PFI
pfi_values = pfc.calculate_and_save_pfi_values(regr_sex_all_feature, X_test, y_test, y)
pfc.plot_pfi_values(pfi_values)

#find proxies in common between these three methods
# Convert lists to sets
set1 = set(shap_most_important_feature_sex['Feature'])
set2 = set(lime_values.iloc[0][:number_of_most_important_feature])
set3 = set(pfi_values['Feature'][:number_of_most_important_feature])

# Find common elements using intersection
common_proxies = set1.intersection(set2).intersection(set3)

# Convert the result back to a list (if needed)
common_proxies = list(common_proxies)

'''--------Retraining the model without using sex and proxies found by using SHAP, SP-LIME AND PFI--------'''

features = features_not_used + common_proxies

# Predicting G3 using all the features except G1 and G2 sex and the proxies
features = data.drop(columns=features)
X = features
y = data['G3']  # Predict final grade

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X.head(60), y.head(60), test_size=0.2, random_state=42)

# Train the model
regr_sex_all_feature = SVR().fit(X_train, y_train)
pred = regr_sex_all_feature.predict(X_test)

#accuracy and fairness (add fairness)
print('\n SVR model - predicting G3 without using protected attribute sex and proxies')
print(metrics.mean_squared_error(y_test, pred))





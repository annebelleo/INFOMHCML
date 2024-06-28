from sklearn.model_selection import train_test_split
import pandas as pd
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#https://www.youtube.com/watch?v=USIZ4Jqa-dM&t=262s
#https://stackoverflow.com/questions/76083485/shap-instances-that-have-more-than-one-dimension

data = pd.read_csv('data/student-por.csv', delimiter=';')
# heatmap requires numerical values
data = data.replace(to_replace=['M', 'F'], value=[0,1]) #sex
data = data.replace(to_replace=['GP', 'MS'], value=[0,1]) #school
data = data.replace(to_replace=['A', 'T'], value=[0,1]) #Pstatus
data = data.replace(to_replace=['GT3', 'LE3'], value=[0,1]) #famsize
data = data.replace(to_replace=['U', 'R'], value=[0,1]) #address
data = data.replace(to_replace=['father', 'mother'], value=[0,1]) #guardian
data = data.replace(to_replace=['at_home', 'health', 'other', 'services', 'teacher'], value=[0,1,2,3,4]) #Mjob, Fjob
data = data.replace(to_replace=['course', 'other', 'home', 'reputation'], value=[0,1,2,3]) #reason
data = data.replace(to_replace=['no', 'yes'], value=[0,1]) #fatherd, nursery, higher, famsup, romantic
data.head()


# split data for training and testing (80:20), use all features except predictors and contributors
features = data
X = features.drop(columns=['G3', 'G1', 'G2'])
y = data['G3'] # predict final grade
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# binary classifier: pass if >= 10, else fail

# redefine values for binary classification
y_train_binary = y_train.tolist()
y_test_binary = y_test.tolist()
for i in range(len(y_train_binary)):
    if y_train_binary[i] >= 10:
        y_train_binary[i] = 1
    else:
        y_train_binary[i] = 0
for i in range(len(y_test_binary)):
    if y_test_binary[i] >= 10:
        y_test_binary[i] = 1
    else:
        y_test_binary[i] = 0

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators' : [200,500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion': ['gini', 'entropy']
}

cv_rfc = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)

# Fit the model on the training data
cv_rfc.fit(X_train, y_train)

# Make predictions on the test data
predictions = cv_rfc.predict(X_test)

# Evaluate the model
accuracy = np.mean(predictions == y_test_binary)

explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
exp = explainer(X_test)
shap_values = explainer.shap_values(X_test)

print(X_train.shape)
print(shap_values.shape)

shap.initjs()
#shap.summary_plot(shap_values,X_test)
shap.plots.beeswarm(exp[:,:,1], max_display=20)
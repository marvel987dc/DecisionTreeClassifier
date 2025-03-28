import os
from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
import seaborn as sns
from dask.array.random import random
from intake.source.cache import display
from joblib import dump
from joblib.testing import param
from mkl_random.mklrand import shuffle
from networkx.algorithms.isomorphism import numerical_multiedge_match
from nltk import entropy, precision, recall
from numba.pycc import export
from pandas.core.common import random_state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.util import decode_slice
from win32comext.adsi.demos.scp import verbose
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, \
    RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import graphviz

data_Juan = pd.read_csv('./Data/student-por.csv', delimiter = ';')

#check dtypes
print("Dtypes: ")
print(data_Juan.dtypes, "\n")
#check missing values
print("Missing Values")
print(data_Juan.isnull().sum(), "\n")
#Statistic Of numeric fields (mean, median, mode, min, max, std, var)
print("Statistics of numeric fields: ")
print(data_Juan.describe(), "\n")
#unique values for each categorical column
print("Unique Values: ")
for col in data_Juan.select_dtypes(include='object').columns:
    print(f"{col}: {data_Juan[col].unique()}  \n")

#Creating a new column called pass_Juan which assign 1 if teh condition is true an 0 if the condition is false
data_Juan['pass_Juan'] = (data_Juan['G1'] + data_Juan['G2'] + data_Juan['G3'] >= 35).astype(int)
#droping the previous columns
data_Juan.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)
#confirm the changes
print("First 5 rows after dropping the columns: ")
print(data_Juan.head(), "\n")

#separate features and target
features_Juan = data_Juan.drop('pass_Juan', axis=1) #all columns except target
target_variables_Juan = data_Juan['pass_Juan'] #target column only

#print the class distribution
class_counts = target_variables_Juan.value_counts()
print("Class Distribution: ")
print(class_counts)
print(f"Percentage: {class_counts[1]/len(target_variables_Juan)*100:.1f}% pass vs {class_counts[0]/len(target_variables_Juan)*100:.1f}% fail")

#calculating the imbalance ratio
imbalance_ratio = max(class_counts) / min(class_counts)
print(f"Imbalance Ratio: {imbalance_ratio: .1f} :1 \n")

#create a list with the numeric features and other one with the categorical features
numeric_features_Juan = features_Juan.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features_Juan = features_Juan.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric columns: ", numeric_features_Juan)
print("Categorical columns: ", cat_features_Juan)

#create column transformer
transformer_Juan = ColumnTransformer(
    transformers=[
        #handle all categorical values and convert them into numeric values using One-Hot-Encoding
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features_Juan)
    ],
    remainder='passthrough', #Preserves all the other columns not explicitly transformed
    verbose_feature_names_out=False
)

#Creation of Decision tree classifier
clf_Juan = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=53
)

#parameters verification
print("Decision Tree parameters:")
print(clf_Juan.get_params())

#Pipeline Creation
pipeline_Juan = Pipeline([
    ('preprocessing', transformer_Juan),
    ('classifier', clf_Juan)
])

# Verify the pipeline steps
print("Pipeline steps:")
print(pipeline_Juan.named_steps)

seed = 53

# 11. Split the data (features_Juan and target_variable_Juan from earlier steps)
X_train_Juan, X_test_Juan, y_train_Juan, y_test_Juan = train_test_split(
    features_Juan,
    target_variables_Juan,
    test_size = 0.2,
    random_state = seed,  # my student ID's last two digits (53)
    stratify=target_variables_Juan
)

#verifying the shapes
print(f"Training set: {X_train_Juan.shape[0]} samples")
print(f"Test set: {X_test_Juan.shape[0]} samples")
print("\nClass distribution in y_train:")
print(y_train_Juan.value_counts(normalize=True))
print("\nClass distribution in y_test:")
print(y_test_Juan.value_counts(normalize=True))

#fitting the pipeline to the training data
pipeline_Juan.fit(X_train_Juan, y_train_Juan)

seed=53
#create Stratified k-fold object with shuffling
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# Perform cross-validation
scores = cross_validate(
    estimator=pipeline_Juan,
    X=X_train_Juan,
    y=y_train_Juan,
    cv=cv_strategy,
    scoring='accuracy',
    return_train_score=True,
    n_jobs=1
)

#printing the mean of the 10 fold cross validation
print("\nAverage test accuracy: ", scores['test_score'].mean())
print("Standard Deviation: ", scores['test_score'].std())

#Visualize The decision Tree
#add Graphviz to the system path
os.environ["PATH"] += os.pathsep + r'C:\Anaconda3\Library\bin\graphviz'

#EXTRACT THE TRAINED TREE FROM YOUR PIPELINE
decision_tree = pipeline_Juan.named_steps['classifier']

#export to DOT format
dot_data = export_graphviz(
    decision_tree,
    out_file=None,
    feature_names=pipeline_Juan.named_steps['preprocessing'].get_feature_names_out(),
    class_names=['Fail', 'Pass'],
    filled=True,
    rounded=True,
    special_characters=True,
    proportion=True
)

#Create and render graph
graph = graphviz.Source(dot_data)
graph.render(filename='decision_tree_Juan', format='png', cleanup=True)
print("Decision tree visualization saved as 'decision_tree_Juan.png'")

#display as a pdf
# graph.view()

#calculate training set accuracy
y_train_pred = pipeline_Juan.predict(X_train_Juan)
train_accuracy = accuracy_score(y_train_Juan, y_train_pred)

#calculate test set accuracy
y_test_pred = pipeline_Juan.predict(X_test_Juan)
test_accuracy = accuracy_score(y_test_Juan, y_test_pred)

#print the accuracies for train and test
print(f"\nTraining accuracy: {train_accuracy: .3f}")
print(f"Testing accuracy: {test_accuracy: .3f}")

#Generate prediction
y_pred = pipeline_Juan.predict(X_test_Juan)

#calculating the metrics
accuracy = accuracy_score(y_test_Juan, y_pred)
precision = precision_score(y_test_Juan, y_pred)
recall = recall_score(y_test_Juan, y_pred)
conf_matrix = confusion_matrix(y_test_Juan, y_pred)

print(f"\nAccuracy score: {accuracy: .3f}")
print(f"Precision Score: {precision: .3f}")
print(f"Recall Sore: {recall: .3f}")
print(f"Confusion Matrix: {conf_matrix}")

## 20. Randomized Grid Search for hyperparameter tuning
parameters={
    'classifier__min_samples_split' : range(10,300,20),
    'classifier__max_depth': range(1,30,2),
    'classifier__min_samples_leaf':range(1,15,3)
}

random_search_Juan = RandomizedSearchCV(
    estimator=pipeline_Juan,
    param_distributions=parameters,
    n_iter=7,
    scoring='accuracy',
    cv=5,
    refit=True,
    verbose=3,
    random_state=53,
)

# Fitting the training Data to the grid search object
random_search_Juan.fit(X_train_Juan, y_train_Juan)

#pritn the results
print("Best parameters:", random_search_Juan.best_params_)
print("Best cross-validation accuracy: {:.3f}".format(random_search_Juan.best_score_))
print("Best estimator: ", random_search_Juan.best_estimator_)


best_model = random_search_Juan.best_estimator_

#printing the accuracy score of the tunned model and the classification report
y_pred = best_model.predict(X_test_Juan)
test_accuracy = accuracy_score(y_test_Juan, y_pred)

print(f"Test Accuracy: {test_accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test_Juan, y_pred))

#saving the model using joblib
decision_tree_model = random_search_Juan.best_estimator_.named_steps['classifier']

dump(decision_tree_model, 'decision_tree_model_Juan.pkl')
print("Decision Tree model saved as 'decision_tree_model_Juan.pkl'")
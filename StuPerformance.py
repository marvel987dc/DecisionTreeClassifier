from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
import seaborn as sns
from dask.array.random import random
from networkx.algorithms.isomorphism import numerical_multiedge_match
from nltk import entropy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from win32comext.adsi.demos.scp import verbose
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import  train_test_split

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

print(f"Training set: {X_train_Juan.shape[0]} samples")
print(f"Test set: {X_test_Juan.shape[0]} samples")
print("\nClass distribution in y_train:")
print(y_train_Juan.value_counts(normalize=True))
print("\nClass distribution in y_test:")
print(y_test_Juan.value_csounts(normalize=True))
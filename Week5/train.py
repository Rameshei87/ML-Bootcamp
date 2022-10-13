#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import cross_validate, cross_val_score
import pickle

# Parameters
C = 1.0
nSplits = 5
outputFile = f'model_C={C}.bin'

# Read the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Peparing and cleaning data # Standarize data formats
df.columns = df.columns.str.lower().str.replace(' ','_')
categCols = df.select_dtypes('object').columns.to_list()

for col in categCols:
    df[col] = df[col].str.lower().str.replace(' ','_')

# Correct values and type of variable totalcharges

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')

# Filling missing values with zero
df.totalcharges.fillna(0, inplace=True)

# Make seniorcitizen an object type variable
#df.seniorcitizen = df.seniorcitizen.astype(bool).astype(object)

rows = []
for col in df.columns:
    rows.append([col,df[col].dtype, df[col].unique()]) 
pd.DataFrame(rows, columns=['Feature', 'Type', 'Unique Values'])

targetCol = 'churn'
target = df[targetCol]
data = df.drop(columns=[targetCol])

# Getting numerical and categorical columns

numColSelector = selector(dtype_exclude=object)
ctgColSelector = selector(dtype_include=object)

numericalCols = numColSelector(data)
categoricalCols = ctgColSelector(data)

del numericalCols[0]
categoricalCols.insert(1, 'seniorcitizen')
del categoricalCols[0]

# creating preprocesors

catPreprocessor = OneHotEncoder(handle_unknown="ignore")
numPreprocessor = StandardScaler()

# Transforming the data

preprocessor = ColumnTransformer([
    ('one-hot-encoder', catPreprocessor, categoricalCols)],remainder="passthrough")

# creating the model

model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, C=C))

# Splitting the data

allColumns = numericalCols + categoricalCols
dataTrainFull, dataTest, targetTrainFull, targetTest = train_test_split(
    data[allColumns], target, test_size=0.2, random_state=1)

dataTrain, dataVal, targetTrain, targetVal = train_test_split(
    dataTrainFull, targetTrainFull, test_size=0.25, random_state=1)

_ = model.fit(dataTrain, targetTrain)

# Let's use the train full dataset and calculate AUC

print(f'Calculating validation with C={C}')
_ = model.fit(dataTrainFull, targetTrainFull)
targetPred = model.predict_proba(dataTest)[:,1]

auc = roc_auc_score(targetTest, targetPred)

cv_results = cross_val_score(model, dataTrainFull, targetTrainFull, scoring='roc_auc', cv=nSplits)

for i in range(len(cv_results)):
    print(f'AUC on fold {i} : {cv_results[i]}')

print("The mean cross-validation accuracy of the final model is: "
      f"{cv_results.mean():.3f} +/- {cv_results.std():.3f}")


# Use pickle to save the model

with open(outputFile, 'wb') as f:
    pickle.dump(model, f)

print(f'Model has been saved to {outputFile}')

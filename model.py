import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import pickle

from sample import sampleData

df = pd.read_csv('heart-disease.csv')
df.dropna(inplace=True)

# drop duplicate rows and columns and columns with very low correlation
df.drop_duplicates(inplace=True)

print(df.corr())
# Drop columns that don't correlate strongly to dependent variable
# Since this is intended to be used in a public facing tool, removing some obscure variables should make this easier to use for the public (perhaps at the expense of some accuracy)
df.drop(columns=['totChol','sysBP','diaBP','heartRate','glucose'], inplace=True)

print(df.corr())

df.corr().to_csv('~/Desktop/heart-disease-corr.csv', header=True)

print(df.shape)

print(df['TenYearCHD'].value_counts().reset_index())


# Split into dependent and independent variables
x = df.drop(columns=['TenYearCHD'],axis=1)
y = df['TenYearCHD']

# Balance dataset using SMOTE
over = SMOTE(sampling_strategy=0.8)
under = RandomUnderSampler(sampling_strategy=0.8)

steps=[("o",over),("u",under)]
pipeline = Pipeline(steps=steps)

x_smote, y_smote = pipeline.fit_resample(x, y)

balanced_df = pd.concat([pd.DataFrame(x_smote), pd.DataFrame(y_smote)], axis=1)
balanced_df.columns = ['sex','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','BMI','TenYearCHD']

# Send the balanced dataframe to CSV for use elsewhere in our application
balanced_df.corr().to_csv("~/Desktop/balanced-corr.csv")

print(balanced_df['TenYearCHD'].value_counts().reset_index())

x_balanced = balanced_df.drop(columns=['TenYearCHD'],axis=1)
y_balanced= balanced_df["TenYearCHD"]

print(y_balanced.head())


# Split dataframe into test and train sets
x_test, x_train, y_test, y_train = train_test_split(x_balanced, y_balanced, test_size=0.3, random_state=48)

# Our model object
lr = LogisticRegression(max_iter=len(x_train.values))
lr.fit(x_train.values, y_train)

y_pred_lr = lr.predict(x_test.values)


# print("score: %s" % log_reg.score(x_test.values, y_test))
print("lr score: %s" % lr.score(x_test, y_test))
print("lr F1: %s" % f1_score(y_test, y_pred_lr))

cm = confusion_matrix(y_test, y_pred_lr)
print(cm)

# Write pickle file of model object
with open("lr.pkl", "wb") as file:
    model = pickle.dump(lr, file)


for i in range(0, len(sampleData)):
    sampleData[i] = np.array(sampleData[i]).reshape(1,-1)
    print(lr.predict(sampleData[i]))
    print(lr.predict_proba(sampleData[i]))



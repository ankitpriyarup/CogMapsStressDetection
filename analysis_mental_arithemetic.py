import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2
from numpy import mean, std


def process(df):
    X = df.drop(['STATE'], axis='columns')
    Y = df['STATE']
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    print('Running KNN...')
    scores_KNN = cross_val_score(KNeighborsClassifier(n_neighbors=1), X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy (KNN): %.6f (%.6f)' % (mean(scores_KNN), std(scores_KNN)))


if __name__ == "__main__":
    all_files = []
    for subject in os.listdir('data_mental_arithemetic/csv_inputs/'):
        print("Added " + subject)
        df = pd.read_csv('data_mental_arithemetic/csv_inputs/' + subject)
        df['STATE'] = df.apply(lambda row: '_2' in subject, axis=1)
        all_files.append(df)

    combined_csv = pd.concat([f for f in all_files ])
    print(combined_csv)
    process(combined_csv)


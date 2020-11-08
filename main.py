import csv
import os
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from numpy import mean, std


def get_score(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)


def process(subject_name, band):
    colToUse = []
    colToUse.append("PHASE")
    for b in band:
        colToUse.append("POW." + b + ".Alpha")
        colToUse.append("POW." + b + ".BetaL")
        colToUse.append("POW." + b + ".BetaH")
        colToUse.append("POW." + b + ".Theta")
    df = pd.read_csv('data_processed/' + subject_name, usecols=colToUse)
    df['STATE'] = df.apply(lambda row: row.PHASE == '3-4' or row.PHASE == '5-6' or row.PHASE == '7-8', axis=1)
    for b in band:
        df[b + ':ALPHA/BETA_L'] = df["POW." + b + ".Alpha"] / df["POW." + b + ".BetaL"]
        df[b + ':ALPHA/BETA_H'] = df["POW." + b + ".Alpha"] / df["POW." + b + ".BetaH"]
        df[b + ':THETA/ALPHA'] = df["POW." + b + ".Theta"] / df["POW." + b + ".Alpha"]
    del df['PHASE']
    for b in band:
        del df["POW." + b + ".Alpha"]
        del df["POW." + b + ".BetaL"]
        del df["POW." + b + ".BetaH"]
        del df["POW." + b + ".Theta"]
    X = df.drop(['STATE'], axis='columns')
    Y = df['STATE']

    #### Standard 25% test and 75% training
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
    # print("Gaussian SVM: " + str(get_score(SVC(kernel='rbf', gamma='auto', C=30), X_train, X_test, Y_train, Y_test)))
    # print("KNN: " + str(get_score(KNeighborsClassifier(n_neighbors=1), X_train, X_test, Y_train, Y_test)))

    #### 10-Cross Fold
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores_gaussianSVM = cross_val_score(SVC(kernel='rbf', gamma='auto', C=10), X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
    scores_KNN = cross_val_score(KNeighborsClassifier(n_neighbors=2), X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy (Gaussian SVM): %.6f (%.6f)' % (mean(scores_gaussianSVM), std(scores_gaussianSVM)))
    print('Accuracy (KNN): %.6f (%.6f)' % (mean(scores_KNN), std(scores_KNN)))


if __name__ == "__main__":
    #### Pick maxCnt csv(s), merge them together and then train on merged
    # all_files = []
    # maxCnt = 5
    # for subject in os.listdir('data_processed'):
    #     if "EEG" in subject and maxCnt > 0:
    #         all_files.append('data_processed/' + subject)
    #         maxCnt = maxCnt-1
    # combined_csv = pd.concat([pd.read_csv(f) for f in all_files ])
    # combined_csv.to_csv("data_processed/Combined.csv", index=False, encoding='utf-8-sig')
    # print("Performing Analysis!")
    # process('Combined.csv', ['AF3', 'F3', 'F4', 'F7', 'F8'])

    #### Every subject seperately
    for subject in os.listdir('data_processed'):
        if "EEG" in subject:
            print("Subject: " + subject)
            process(subject, ['AF3', 'F3', 'F4', 'F7', 'F8'])

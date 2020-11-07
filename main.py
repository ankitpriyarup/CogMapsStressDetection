import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np


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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    model = SVC(kernel='rbf', gamma='auto', C=2.5)
    model.fit(X_train, Y_train)
    print("subject: " + str(subject_name) + ", band: " + str(band) + ", score:" + str(model.score(X_test, Y_test)))


if __name__ == "__main__":
    for subject in os.listdir('data_processed'):
        if "EEG" in subject:
            process(subject, ['AF3', 'F3', 'F4', 'F7', 'F8'])

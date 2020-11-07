import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np


def process(subject_name, band):
    df = pd.read_csv('data_processed/' + subject_name + '_EEG.csv', usecols=["PHASE", "POW." + band + ".Alpha", "POW." + band + ".BetaL", "POW." + band + ".BetaH", "POW." + band + ".Theta"])
    df['STATE'] = df.apply(lambda row: row.PHASE == '3-4' or row.PHASE == '5-6' or row.PHASE == '7-8', axis=1)
    df['ALPHA/BETA_L'] = df["POW." + band + ".Alpha"] / df["POW." + band + ".BetaL"]
    df['ALPHA/BETA_H'] = df["POW." + band + ".Alpha"] / df["POW." + band + ".BetaH"]
    df['THETA/BETA_L'] = df["POW." + band + ".Theta"] / df["POW." + band + ".BetaL"]
    df['THETA/BETA_H'] = df["POW." + band + ".Theta"] / df["POW." + band + ".BetaH"]
    del df['PHASE']
    del df["POW." + band + ".Alpha"]
    del df["POW." + band + ".BetaL"]
    del df["POW." + band + ".BetaH"]
    del df["POW." + band + ".Theta"]
    X = df.drop(['STATE'], axis='columns')
    Y = df['STATE']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    model = SVC(kernel='rbf', gamma='auto', C=2.5)
    model.fit(X_train, Y_train)
    print("subject: " + str(subject_name) + ", band: " + str(band) + ", score:" + str(model.score(X_test, Y_test)))


if __name__ == "__main__":
    for band in ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]:
        process("Subject10", band)

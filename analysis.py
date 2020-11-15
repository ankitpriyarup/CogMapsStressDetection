import csv, os, statistics, warnings, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
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

pio.orca.config.executable = 'C:/Users/ankitpriyarup/AppData/Local/Programs/orca/orca.exe'
pio.orca.config.save()
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
CHOSEN_FEATURES = [ "EDA", "HR", "TEMP" ]

DISCARDED_FEATURES = [ "AF3_MINIMUM", "AF3_MAXIMUM", "AF3_MEAN", "AF3_STANDARD_DEVIATION",
                       "F7_MINIMUM", "F7_MAXIMUM", "F7_MEAN", "F7_STANDARD_DEVIATION",
                       "F3_MINIMUM", "F3_MAXIMUM", "F3_MEAN", "F3_STANDARD_DEVIATION",
                       "FC5_MINIMUM", "FC5_MAXIMUM", "FC5_MEAN", "FC5_STANDARD_DEVIATION",
                       "T7_MINIMUM", "T7_MAXIMUM", "T7_MEAN", "T7_STANDARD_DEVIATION",
                       "P7_MINIMUM", "P7_MAXIMUM", "P7_MEAN", "P7_STANDARD_DEVIATION",
                       "O1_MINIMUM", "O1_MAXIMUM", "O1_MEAN", "O1_STANDARD_DEVIATION",
                       "O2_MINIMUM", "O2_MAXIMUM", "O2_MEAN", "O2_STANDARD_DEVIATION",
                       "P8_MINIMUM", "P8_MAXIMUM", "P8_MEAN", "P8_STANDARD_DEVIATION",
                       "T8_MINIMUM", "T8_MAXIMUM", "T8_MEAN", "T8_STANDARD_DEVIATION",
                       "FC6_MINIMUM", "FC6_MAXIMUM", "FC6_MEAN", "FC6_STANDARD_DEVIATION",
                       "F4_MINIMUM", "F4_MAXIMUM", "F4_MEAN", "F4_STANDARD_DEVIATION",
                       "F8_MINIMUM", "F8_MAXIMUM", "F8_MEAN", "F8_STANDARD_DEVIATION",
                       "AF4_MINIMUM", "AF4_MAXIMUM", "AF4_MEAN", "AF4_STANDARD_DEVIATION",
                       "TIME", "EDA", "HR", "TEMP" ]

def get_score(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)


def generate_accuracy_and_heatmap(model, x, y):
#     cm = confusion_matrix(y,model.predict(x))
#     sns.heatmap(cm,annot=True,fmt="d")
    ac = accuracy_score(y,model.predict(x))
    f_score = f1_score(y,model.predict(x))
    print('Accuracy is: ', ac)
    print('F1 score is: ', f_score)
    print ("\n")
    print (pd.crosstab(pd.Series(model.predict(x), name='Predicted'),
                       pd.Series(y['Outcome'],name='Actual')))
    return 1


features_val = {}
def process(subject_name):
    df = pd.read_csv('data_processed/' + subject_name)
    df['STATE'] = df.apply(lambda row: row.PHASE == '3 -> 4' or row.PHASE == '5 -> 6' or row.PHASE == '7 -> 8', axis=1)
    del df['PHASE']
    
    for col in df.columns:
        # if col in DISCARDED_FEATURES:
        #     del df[col]
        if col not in CHOSEN_FEATURES and col != 'STATE':
            del df[col]
    X = df.drop(['STATE'], axis='columns')
    Y = df['STATE']

    #### Feature Selection
    # numerical_feature_columns = list(df._get_numeric_data().columns)
    # target = 'STATE'
    # k = 42 # number of variables for heatmap
    # cols = df[numerical_feature_columns].corr().nlargest(k, target)[target].index
    # cm = df[cols].corr()
    # plt.figure(figsize=(30,18))
    # heatmap = sns.heatmap(cm, annot=True, cmap = 'viridis')
    # heatmap.figure.savefig("observations/heatmap_" + subject_name.replace('.csv', '') + ".jpg")
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
    # select_feature = SelectKBest(chi2, k=5).fit(X_train, Y_train)
    # selected_features_df = pd.DataFrame({'Feature':list(X_train.columns), 'Scores':select_feature.scores_})
    # cur_line = 0
    # while cur_line < k:
    #     feature_name = str(selected_features_df['Feature'][cur_line])
    #     if feature_name in features_val:
    #         features_val[feature_name] = features_val[feature_name] + selected_features_df['Scores'][cur_line]
    #     else:
    #         features_val[feature_name] = selected_features_df['Scores'][cur_line]
    #     cur_line = cur_line+1
    # rfecv = RFECV(RandomForestClassifier(), scoring="accuracy")
    # rfecv = rfecv.fit(X_train, Y_train)
    # selected_rfecv_features = pd.DataFrame({'Feature':list(X_train.columns), 'Ranking':rfecv.ranking_})
    # print(selected_rfecv_features.sort_values(by='Ranking'))

    #### Standard 25% test and 75% training
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
    # print("KNN: " + str(get_score(KNeighborsClassifier(n_neighbors=1), X_train, X_test, Y_train, Y_test)))
    # print("Gaussian SVM: " + str(get_score(SVC(kernel='rbf', gamma='auto', C=2.5), X_train, X_test, Y_train, Y_test)))

    #### 10-Cross Fold
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    print('Running KNN...')
    scores_KNN = cross_val_score(KNeighborsClassifier(n_neighbors=1), X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy (KNN): %.6f (%.6f)' % (mean(scores_KNN), std(scores_KNN)))

    # print('Running Gaussian Naive Bayes...')
    # scores_GaussianNB = cross_val_score(GaussianNB(), X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
    # print('Accuracy (KNN): %.6f (%.6f)' % (mean(scores_GaussianNB), std(scores_GaussianNB)))

    # print('Running Random Forest...')
    # scores_RandomForest = cross_val_score(RandomForestClassifier(), X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
    # print('Accuracy (KNN): %.6f (%.6f)' % (mean(scores_RandomForest), std(scores_RandomForest)))

    # print('Running Gaussian SVM...')
    # scores_gaussianSVM = cross_val_score(SVC(kernel='rbf', gamma='auto', C=2.5), X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
    # print('Accuracy (Gaussian SVM): %.6f (%.6f)' % (mean(scores_gaussianSVM), std(scores_gaussianSVM)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'a':
        # Pick maxCnt csv(s), merge them together and then train on merged
        all_files = []
        for subject in os.listdir('data_processed'):
            print("Added " + subject)
            all_files.append('data_processed/' + subject)
        combined_csv = pd.concat([pd.read_csv(f) for f in all_files ])
        combined_csv.to_csv("data_processed/Combined.csv", index=False, encoding='utf-8-sig')
        print("Performing Analysis!")
        process('Combined.csv')
    else:
        # Every subject seperately
        for subject in os.listdir('data_processed'):
            print("Subject: " + subject)
            process(subject)
        # selectedFeatures = pd.DataFrame(list(sorted(features_val.items(), key=lambda x:x[1], reverse=True)))
        # selectedFeatures.to_csv('observations/chi2test.csv')
import csv, os, statistics, warnings, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.svm import SVC, SVR
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from numpy import mean, std

pio.orca.config.executable = 'C:/Users/ankitpriyarup/AppData/Local/Programs/orca/orca.exe'
pio.orca.config.save()
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
CHOSEN_FEATURES = [ "O2_POW_THETA_RELATIVE", "FC6_POW_ALPHA_BY_BETA_H", "FC5_POW_BETA_H_RELATIVE", "FC6_POW_ALPHA_BY_BETA_L", "T8_SPECTRAL_ENTROPY", "T8_POW_BETA_H_RELATIVE",
                    "T8_POW_ALPHA_RELATIVE", "T8_POW_THETA_RELATIVE", "T7_POW_THETA_RELATIVE", "T8_POW_THETA_BY_ALPHA", "T8_POW_ALPHA_BY_BETA_H", "T7_POW_BETA_H_RELATIVE", "T8_POW_ALPHA_BY_BETA_L",
                    "T8_MOBILITY", "FC6_POW_THETA_BY_ALPHA", "P8_SPECTRAL_ENTROPY", "P7_POW_ALPHA_BY_BETA_H", "P8_POW_BETA_L_RELATIVE", "P8_POW_ALPHA_RELATIVE", "P8_POW_THETA_RELATIVE", "P8_POW_THETA_BY_ALPHA",
                    "P7_POW_BETA_H_RELATIVE", "P8_POW_ALPHA_BY_BETA_H", "P8_POW_ALPHA_BY_BETA_L", "O2_SPECTRAL_ENTROPY", "O2_POW_BETA_H_RELATIVE", "O1_POW_ALPHA_BY_BETA_H", "O1_POW_ALPHA_RELATIVE",
                    "O1_POW_BETA_L_RELATIVE", "O1_POW_BETA_H_RELATIVE", "P8_POW_BETA_H_RELATIVE", "FC5_POW_ALPHA_BY_BETA_H", "FC5_POW_THETA_RELATIVE", "FC6_POW_THETA_RELATIVE", "AF4_POW_BETA_L_RELATIVE",
                    "AF4_POW_THETA_RELATIVE", "AF3_POW_ALPHA_BY_BETA_H", "AF4_POW_THETA_BY_ALPHA", "AF4_POW_ALPHA_BY_BETA_L", "AF3_POW_BETA_L_RELATIVE", "AF3_POW_BETA_H_RELATIVE", "F8_POW_BETA_H_RELATIVE",
                    "F8_POW_ALPHA_RELATIVE", "F7_POW_ALPHA_BY_BETA_L", "F7_POW_ALPHA_BY_BETA_H", "F8_POW_ALPHA_BY_BETA_H", "FC5_POW_ALPHA_BY_BETA_L", "F4_SPECTRAL_ENTROPY", "F8_POW_ALPHA_BY_BETA_L",
                    "F4_POW_BETA_H_RELATIVE", "FC6_POW_ALPHA_RELATIVE", "FC6_POW_BETA_L_RELATIVE", "F3_POW_BETA_H_RELATIVE", "F7_POW_BETA_H_RELATIVE", "FC6_POW_BETA_H_RELATIVE", "F3_POW_THETA_RELATIVE",
                    "F3_POW_BETA_L_RELATIVE", "F3_POW_ALPHA_BY_BETA_H", "F4_POW_ALPHA_BY_BETA_L", "F4_POW_ALPHA_BY_BETA_H", "F4_POW_BETA_L_RELATIVE", "FC6_SPECTRAL_ENTROPY", "O2_POW_ALPHA_BY_BETA_H",

                    "AF3_POW_THETA_RELATIVE", "FC5_SPECTRAL_ENTROPY", "FC5_POW_BETA_L_RELATIVE", "O2_POW_ALPHA_RELATIVE", "F7_POW_ALPHA_RELATIVE", "O2_POW_BETA_L_RELATIVE", "O1_POW_THETA_RELATIVE",
                    "F8_POW_THETA_BY_ALPHA", "AF4_SPECTRAL_ENTROPY", "T7_POW_ALPHA_RELATIVE", "T7_POW_ALPHA_BY_BETA_H", "AF3_POW_ALPHA_BY_BETA_L", "T8_POW_BETA_L_RELATIVE", "F8_POW_BETA_L_RELATIVE",
                    "F3_POW_THETA_BY_ALPHA", "F7_POW_THETA_RELATIVE", "O1_POW_ALPHA_BY_BETA_L", "F4_POW_ALPHA_RELATIVE", "T8_COMPLEXITY", "F7_POW_BETA_L_RELATIVE", "F4_POW_THETA_RELATIVE",
                    "F4_POW_THETA_BY_ALPHA", "P7_POW_ALPHA_BY_BETA_L", "F8_POW_THETA_RELATIVE", "T7_POW_ALPHA_BY_BETA_L", "P7_POW_THETA_BY_ALPHA", "AF4_POW_ALPHA_BY_BETA_H", "O2_POW_ALPHA_BY_BETA_L",

                    "AF4_POW_ALPHA_RELATIVE", "T7_POW_BETA_L_RELATIVE", "AF4_POW_BETA_H_RELATIVE", "P8_MOBILITY", "P7_POW_ALPHA_RELATIVE", "FC5_POW_ALPHA_RELATIVE", "O1_POW_THETA_BY_ALPHA",
                    "O2_POW_THETA_BY_ALPHA", "AF3_POW_ALPHA_RELATIVE", "P7_POW_THETA_RELATIVE", "AF3_SPECTRAL_ENTROPY", "F3_POW_ALPHA_RELATIVE", "F3_SPECTRAL_ENTROPY", "F3_POW_ALPHA_BY_BETA_L",
                    "T7_SPECTRAL_ENTROPY", "P8_COMPLEXITY", "P7_MOBILITY", "F7_POW_THETA_BY_ALPHA", "P7_POW_BETA_L_RELATIVE", "F7_SPECTRAL_ENTROPY", "F8_SPECTRAL_ENTROPY", "P7_SPECTRAL_ENTROPY",
                    "T7_POW_THETA_BY_ALPHA", "AF3_POW_THETA_BY_ALPHA", "FC5_POW_THETA_BY_ALPHA", "AF4_COMPLEXITY", "P7_COMPLEXITY", "O2_MOBILITY", "AF4_MOBILITY", "O1_MOBILITY", "F4_COMPLEXITY",
                    "T7_MOBILITY", "T7_COMPLEXITY", "FC6_MOBILITY", "F8_MOBILITY", "FC6_COMPLEXITY", "F4_MOBILITY", "F3_COMPLEXITY", "O2_COMPLEXITY", "FC5_MOBILITY", "O1_COMPLEXITY", "F7_MOBILITY",
                    "FC5_COMPLEXITY", "F8_COMPLEXITY", "F3_MOBILITY", "F7_COMPLEXITY", "AF3_MOBILITY", "AF3_COMPLEXITY"

 ]

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
    for i in [116]:
    # for i in range(1, len(CHOSEN_FEATURES)+1):
        print("Taking " + str(i) + " features...")
        df = pd.read_csv('data_processed/' + subject_name)
        df['STATE'] = df.apply(lambda row: row.PHASE == '3 -> 4' or row.PHASE == '5 -> 6' or row.PHASE == '7 -> 8', axis=1)
        del df['PHASE']

        df = df.sort_values('STATE', ascending=False)
        baselineCount = len(df[df.STATE == 0])
        stressCount = len(df[df.STATE == 1])

        # Delete initial rows
        df = df.iloc[(stressCount - baselineCount):]

        # Uniformly delete rows from stress to balance it
        # toRemove = []
        # cluster = math.ceil(stressCount // (stressCount - baselineCount))
        # for i in range(0, stressCount):
        #     if i % cluster == 0:
        #         toRemove.append(i)
        # df = df.drop(df.index[toRemove])

        baselineCount = len(df[df.STATE == 0])
        stressCount = len(df[df.STATE == 1])
        print("Baseline count: " + str(baselineCount))
        print("Stress count: " + str(stressCount))

        for col in df.columns:
            # if col in DISCARDED_FEATURES:
            #     del df[col]
            if col not in CHOSEN_FEATURES[:i] and col != 'STATE':
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
        # fin = selected_rfecv_features.sort_values(by='Ranking')
        # fin.to_csv('feature_selected.csv')

        #### Standard 25% test and 75% training
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
        # print("KNN: " + str(get_score(KNeighborsClassifier(n_neighbors=1), X_train, X_test, Y_train, Y_test)))
        # print("Gaussian SVM: " + str(get_score(SVC(kernel='rbf', gamma='auto', C=2.5), X_train, X_test, Y_train, Y_test)))


        #### HyperParameter Tuning
        # KNN
        # hyperparameters = dict(leaf_size=list(range(1, 10)), n_neighbors=list(range(1, 6)), p=[1, 2])
        # clf = GridSearchCV(KNeighborsClassifier(), hyperparameters)
        # best_model = clf.fit(X, Y)
        # print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
        # print('Best p:', best_model.best_estimator_.get_params()['p'])
        # print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

        # Random Forest
        # hyperparameters = dict(max_features=["auto", "sqrt", "log2"], n_estimators=list(range(100, 300, 10)))
        # clf = GridSearchCV(RandomForestClassifier(), hyperparameters)
        # best_model = clf.fit(X, Y)
        # print('Best max_features:', best_model.best_estimator_.get_params()['max_features'])
        # print('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])

        #### 10-Cross Fold
        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        models = [KNeighborsClassifier(n_neighbors=1), RandomForestClassifier(), GaussianNB(), SVC(kernel='rbf', gamma='auto')]
        names = ['KNN', "Random Forest", "Gaussian Naive Bayes", "Gaussian SVM"]

        for model, name in zip(models, names):
            print('Running ' + name + '...')
            start = time.time()
            for score in ["accuracy", "f1"]:
                print(score + ': ' + str(cross_val_score(model, X, Y, scoring=score, cv=cv).mean()))
            print('Time Taken: ' + str(time.time() - start))
            print('----------------')


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'a':
        # Pick all csv(s), merge them together and then train on merged
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
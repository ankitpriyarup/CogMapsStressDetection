import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from scipy import stats
from scipy.signal import savgol_filter
import os, sys, time, datetime, csv, math, shutil, random, matplotlib
from numpy import interp
from sklearn.datasets import make_blobs
from sklearn.metrics import auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix


def draw_cv_roc_curve(classifier, cv, X, y, title='ROC Curve'):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train],
                                 y.iloc[train]).predict_proba(X.iloc[test])
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr,
                 tpr,
                 lw=1,
                 alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1],
             linestyle='--',
             lw=2,
             color='r',
             label='Luck',
             alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr,
             mean_tpr,
             color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2,
             alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr,
                     tprs_lower,
                     tprs_upper,
                     color='grey',
                     alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def processValue(signal, applySavgol):
    sig = []
    for i in signal:
        for j in i:
            sig.append(j)

    if applySavgol == True:
        sig = savgol_filter(sig, 3, 2)

    maximum = np.amax(sig)
    minimum = np.amin(sig)
    mean = np.mean(sig)
    median = np.median(sig)
    mode = np.mean(stats.mode(sig).mode)
    standard_deviation = np.std(sig)
    first_derrivative = np.mean(np.diff(sig))
    # return [mean]
    return [
        maximum, minimum, mean, median, mode, standard_deviation,
        first_derrivative
    ]
    # return [maximum, minimum, mean, standard_deviation]


def performPreprocessing(location, subject_name, start_time):
    SAMPLING_FREQUENCY = 64
    STEPS = SAMPLING_FREQUENCY // 16
    DURATIONS = [
        3 * 60 + 2, 8 * 60 + 32, 17 * 60 + 6, 18 * 60 + 5, 30 * 60 + 7,
        31 * 60 + 37, 40 * 60 + 9, 45 * 60 + 40
    ]
    points = [(start_time).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[0])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[1])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[2])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[3])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[4])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[5])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[6])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[7])).time()]

    # Sampling frequency is: 4, 1, 4, 64 Hz respectively, upsampling everything to 64 Hz
    eda, hr, skt, bvp = [], [], [], []
    data_eda, data_hr, data_skt, data_bvp = pd.read_csv(
        location + "/WatchData/EDA.csv"), pd.read_csv(
            location + "/WatchData/HR.csv"), pd.read_csv(
                location +
                "/WatchData/TEMP.csv"), pd.read_csv(location +
                                                    "/WatchData/BVP.csv")
    for val in data_eda.values.tolist()[41:]:
        for i in range(0, SAMPLING_FREQUENCY // 4):
            eda.append(val)
    for val in data_hr.values.tolist()[1:]:
        for i in range(0, SAMPLING_FREQUENCY):
            hr.append(val)
    for val in data_skt.values.tolist()[41:]:
        for i in range(0, SAMPLING_FREQUENCY // 4):
            skt.append(val)
    for val in data_bvp.values.tolist()[641:]:
        for i in range(0, SAMPLING_FREQUENCY // 64):
            bvp.append(val)

    labels, rows, pos = [], [], 0
    labels.append('PHASE')
    for sig in ['EDA', 'HR', 'SKT', 'BVP']:
        # for feature in ['MEAN']:
        # for feature in ['MAX', 'MIN', 'MEAN', 'SD']:
        for feature in ['MAX', 'MIN', 'MEAN', 'MEDIAN', 'MODE', 'SD', 'DIFF']:
            labels.append(sig + '_' + feature)
    cur_time = datetime.datetime.strptime(
        str(
            time.strftime('%H:%M:%S',
                          time.localtime(float(data_hr.columns[0])))),
        '%H:%M:%S')
    for i in range(0, len(eda), STEPS):
        cur = []
        if pos == 0 or (pos < len(points) and cur_time.time() == points[pos]):
            pos = pos + 1
        if i + STEPS >= len(eda) or pos >= 9:
            break
        cur.append(str(pos) + ' -> ' + str(pos + 1))
        for j in [
                processValue(eda[i:(i + STEPS)], True),
                processValue(hr[i:(i + STEPS)], False),
                processValue(skt[i:(i + STEPS)], True),
                processValue(bvp[i:(i + STEPS)], False)
        ]:
            for k in j:
                cur.append(str(k))
        rows.append(cur)
        cur_time = cur_time + datetime.timedelta(seconds=1)
        rows.append(cur)

    with open('data_processed_new/' + subject_name + '.csv', 'w',
              newline='') as file:
        write = csv.writer(file)
        write.writerow(labels)
        write.writerows(rows)


# FEATURE_PREFERENCE = ['EDA_MAX', 'BVP_MEAN', 'BVP_MIN', 'BVP_MAX', 'SKT_MODE', 'SKT_MEDIAN', 'SKT_MEAN', 'SKT_MIN', 'SKT_MAX', 'HR_MODE', 'BVP_SD', 'HR_MEAN', 'HR_MIN', 'HR_MAX', 'EDA_MODE', 'EDA_MEDIA', 'EDA_MEAN', 'EDA_MIN', 'HR_MEDIAN', 'BVP_MODE', 'BVP_MEDIAN', 'SKT_SD', 'EDA_SD', 'HR_SD']
FEATURE_PREFERENCE = ['EDA', 'HR', 'SKT', 'BVP']
OPTIMIZE_MODE = False
cur_best, best_len, hash_map = 0, 0, []

from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns


def performAnalysis(subject_name):
    global hash_map
    global cur_best
    global best_len

    if (OPTIMIZE_MODE == False):
        print("Started Analysis")
    else:
        random.shuffle(FEATURE_PREFERENCE)

    chosen = 0
    indV = 0
    # for chosen in range(0, len(FEATURE_PREFERENCE)):
    # for chosen_feature in (['BVP'], ['HR', 'EDA'], ['EDA', 'SKT', 'HR'], ['EDA', 'SKT', 'HR', 'BVP']):

    featurePool = [['EDA'], ['HR'], ['SKT'], ['BVP'], ['EDA', 'HR'],
                   ['EDA', 'SKT'], ['EDA', 'BVP'], ['HR', 'SKT'],
                   ['HR', 'BVP'], ['SKT', 'BVP'], ['EDA', 'HR', 'SKT'],
                   ['EDA', 'HR', 'BVP'], ['EDA', 'SKT', 'BVP'],
                   ['HR', 'SKT', 'BVP'], ['EDA', 'HR', 'SKT', 'BVP']]

    for chosen_feature in (featurePool):
        indV = indV + 1
        if (OPTIMIZE_MODE == False):
            # print("Picking top " + str(chosen + 1) + " feature(s)")
            print(chosen_feature)
        else:
            key = ''
            taken = FEATURE_PREFERENCE[:(chosen + 1)]
            taken.sort()
            for x in taken:
                key = key + x[0]
            if key in hash_map:
                continue
            else:
                hash_map.append(key)

        df = pd.read_csv('data_processed_new/' + subject_name)
        df['STATE'] = df.apply(
            lambda row: 1
            if row.PHASE == '3 -> 4' or row.PHASE == '7 -> 8' else
            (2 if row.PHASE == '5 -> 6' else 0),
            axis=1)
        del df['PHASE']

        print(df.corr(method='spearman'))
        numerical_feature_columns = list(df._get_numeric_data().columns)
        target = 'STATE'
        k = 29  # number of variables for heatmap
        cols = df[numerical_feature_columns].corr().nlargest(
            k, target)[target].index

        print(cols)

        cm = df[cols].corr()
        plt.figure(figsize=(48, 30))
        heatmap = sns.heatmap(cm, annot=True, cmap='viridis')
        heatmap.figure.savefig("heatmap.jpg")

        baselineCount = len(df[df.STATE == 0])
        stressCount = len(df[df.STATE == 1])
        highStressCount = len(df[df.STATE == 2])
        print("Baseline count: " + str(baselineCount))
        print("Stress count: " + str(stressCount))
        print("High Stress count: " + str(highStressCount))

        #### Validation
        features = []
        for col in df.columns:
            if col != 'STATE':
                safe = False
                for item in chosen_feature:
                    # for item in FEATURE_PREFERENCE[:(chosen + 1)]:
                    if item in col:
                        safe = True
                        break
                if safe == False:
                    del df[col]
                else:
                    features.append(col)
            else:
                features.append(col)
        if (OPTIMIZE_MODE == False):
            print(features)
        X = df.drop(['STATE'], axis='columns')  # NEVER COMMENT BITCH
        Y = df['STATE']  # NEVER COMMENT BITCH
        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        models = [
            GradientBoostingClassifier(n_estimators=100,
                                       learning_rate=1.0,
                                       max_depth=1,
                                       random_state=0),
            LinearDiscriminantAnalysis(),
            AdaBoostClassifier(n_estimators=100, random_state=0),
            KNeighborsClassifier(n_neighbors=1),
            RandomForestClassifier(),
        ]
        names = ["GB", "LDA", "ADA", "KNN", "RF"]
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.25)
        for model, name in zip(models, names):
            if (OPTIMIZE_MODE == False):
                print('Running ' + name + '...')
            for score in ["accuracy", "roc_auc_ovr"]:
                cur_score = cross_val_score(model, X, Y, scoring=score,
                                            cv=cv).mean()
                if (OPTIMIZE_MODE == False):
                    print(score + ': ' + str(cur_score))
                else:
                    if cur_score > cur_best or (cur_score == cur_best
                                                and len(features) < best_len):
                        print('NEW BEST!')
                        cur_best = cur_score
                        best_len = len(features)
                    print(str(cur_score))
                    print(FEATURE_PREFERENCE[:(chosen + 1)])
                    print(features)
                    print('----------------')
            model.fit(X_train, Y_train)
            predicted_labels = model.predict(X_test)
            disp = plot_confusion_matrix(model,
                                         X_test,
                                         Y_test,
                                         cmap=plt.cm.Blues)
            plt.savefig('results_new/confusion_' + str(indV) + '_' + name +
                        '.png')

            if (OPTIMIZE_MODE == False):
                print('----------------')

        #### Feature Selection
        # print('Running Feature Selection')

        # reg = LassoCV()
        # reg.fit(X, Y)
        # print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
        # print("Best score using built-in LassoCV: %f" %reg.score(X,Y))
        # coef = pd.Series(reg.coef_, index = X.columns)

        # print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

        # imp_coef = coef.sort_values()
        # matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
        # imp_coef.plot(kind = "barh")
        # plt.title("Feature importance using Lasso Model")
        # plt.savefig('lasso_pick.png')

        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
        # rfecv = RFECV(RandomForestClassifier(), scoring="accuracy")
        # rfecv = rfecv.fit(X_train, Y_train)
        # selected_rfecv_features = pd.DataFrame({'Feature':list(X_train.columns), 'Ranking':rfecv.ranking_})
        # fin = selected_rfecv_features.sort_values(by='Ranking')
        # fin.to_csv('feature_selected_new.csv')


if __name__ == "__main__":
    # shutil.rmtree('data_processed_new')
    # os.mkdir("data_processed_new")

    # print("Performing preprocessing")
    # for subject in os.listdir('data'):
    #     if "Subject5" in subject:
    #         continue
    #     infile = open('data/' + subject + '/Log.txt', 'r')
    #     start_time = datetime.datetime.strptime(
    #         infile.readline().replace('\n', ''), '%H:%M:%S')
    #     print('Performing on ' + subject + ' start_time=' +
    #           str(start_time.time()))
    #     performPreprocessing('data/' + subject, subject, start_time)

    print("Performing analysis")
    all_files, already_combined = [], False
    for subject in os.listdir('data_processed_new'):
        if "Combined" in subject:
            already_combined = True
            break
        all_files.append('data_processed_new/' + subject)
    if already_combined == False:
        combined_csv = pd.concat([pd.read_csv(f) for f in all_files])
        combined_csv.to_csv("data_processed_new/Combined.csv",
                            index=False,
                            encoding='utf-8-sig')

    while True:
        performAnalysis('Combined.csv')
        if OPTIMIZE_MODE == False:
            break

import os
import sys
import csv
import time
import shutil
import datetime
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go


pio.orca.config.executable = 'C:/Users/ankitpriyarup/AppData/Local/Programs/orca/orca.exe'
pio.orca.config.save()
duration_training = 3*60 + 2
duration_relax = 8*60 + 32
duration_control = 17*60 + 6
duration_rest = 18*60 + 5
duration_test = 30*60 + 7
duration_rest2 = 31*60 + 37
duration_control2 = 40*60 + 9
duration_complete = 45*60 + 40

def plot(axis, minhr, maxhr, subject_name, start_time):
    df = pd.read_csv('data_processed/' + subject_name + '_ECG.csv', usecols=[axis, 'TIME'])
    fig = px.line(df, x='TIME', y=axis)
    fig.add_shape(type="line",
        x0 = str(start_time.time()), y0 = minhr, x1 = str(start_time.time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_training)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_training)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_relax)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_relax)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_control)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_control)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_rest)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_rest)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_test)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_test)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_rest2)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_rest2)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_control2)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_control2)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_complete)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_complete)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.write_image("results/" + subject_name + "_" + axis + ".png")

    line = 0
    X = list(df['TIME'])
    Y = list(df[axis])
    point1 = (start_time).time()
    point2 = (start_time + datetime.timedelta(seconds=duration_training)).time()
    point3 = (start_time + datetime.timedelta(seconds=duration_relax)).time()
    point4 = (start_time + datetime.timedelta(seconds=duration_control)).time()
    point5 = (start_time + datetime.timedelta(seconds=duration_rest)).time()
    point6 = (start_time + datetime.timedelta(seconds=duration_test)).time()
    point7 = (start_time + datetime.timedelta(seconds=duration_rest2)).time()
    point8 = (start_time + datetime.timedelta(seconds=duration_control2)).time()
    point9 = (start_time + datetime.timedelta(seconds=duration_complete)).time()
    vals = [0, 0, 0, 0, 0, 0, 0, 0]
    cnt = [0, 0, 0, 0, 0, 0, 0, 0]
    while (line < len(X)):
        cur = datetime.datetime.strptime(X[line], '%H:%M:%S')
        if cur.time() >= point1 and cur.time() < point2:
            vals[0] = vals[0] + float(Y[line])
            cnt[0] = cnt[0] + 1
        if cur.time() >= point2 and cur.time() < point3:
            vals[1] = vals[1] + float(Y[line])
            cnt[1] = cnt[1] + 1
        if cur.time() >= point3 and cur.time() < point4:
            vals[2] = vals[2] + float(Y[line])
            cnt[2] = cnt[2] + 1
        if cur.time() >= point4 and cur.time() < point5:
            vals[3] = vals[3] + float(Y[line])
            cnt[3] = cnt[3] + 1
        if cur.time() >= point5 and cur.time() < point6:
            vals[4] = vals[4] + float(Y[line])
            cnt[4] = cnt[4] + 1
        if cur.time() >= point6 and cur.time() < point7:
            vals[5] = vals[5] + float(Y[line])
            cnt[5] = cnt[5] + 1
        if cur.time() >= point7 and cur.time() < point8:
            vals[6] = vals[6] + float(Y[line])
            cnt[6] = cnt[6] + 1
        if cur.time() >= point8 and cur.time() < point9:
            vals[7] = vals[7] + float(Y[line])
            cnt[7] = cnt[7] + 1
        line = line+1

    for i, val in enumerate(cnt):
        if val > 0:
            vals[i] = vals[i] / val
    legends = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9"]
    newdf = pd.DataFrame(list(zip(legends, vals)), columns=['legends', axis])
    fig2 = px.line(newdf, x='legends', y=axis)
    fig2.write_image("results/[AVG]" + subject_name + "_" + axis + ".png")
    # fig.show()

def perform(location, subject_name, start_time):
    # 4, 1, 4 Hz freq respectively, sampling it uniformly at 4 Hz
    # HR starts at 10 sec delay, so offset others accordingly
    minhr = 1000
    maxhr = -1000
    mineda = 1000
    maxeda = -1000
    mintemp = 1000
    maxtemp = -1000
    fields = ['TIME', 'EDA', 'HR', 'TEMP']
    data = pd.read_csv(location + "/EDA.csv")
    curTime = float(list(data.columns)[0])
    eda = data.values.tolist()[41:]
    data = pd.read_csv(location + "/HR.csv")
    hr = []
    for i in data.values.tolist()[1:]:
        hr.append(i)
        hr.append(i)
        hr.append(i)
        hr.append(i)
    data = pd.read_csv(location + "/TEMP.csv")
    temp = data.values.tolist()[41:]

    toLook = min(len(eda), min(len(hr), len(temp)))

    rows = []
    line = 0
    while line < toLook:
        cur = []
        cur.append(str(time.strftime('%H:%M:%S', time.localtime(curTime))))
        mineda = min(mineda, eda[line][0])
        maxeda = max(maxeda, eda[line][0])
        cur.append(eda[line][0])
        minhr = min(minhr, hr[line][0])
        maxhr = max(maxhr, hr[line][0])
        cur.append(hr[line][0])
        mintemp = min(mintemp, temp[line][0])
        maxtemp = max(maxtemp, temp[line][0])
        cur.append(temp[line][0])
        rows.append(cur)
        line = line+1
        if line % 4 == 0:
            curTime = curTime+1

    with open('data_processed/' + subject_name + '_ECG.csv', 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(fields)
        write.writerows(rows)

    plot('HR', minhr, maxhr, subject_name, start_time)
    plot('EDA', mineda, maxeda, subject_name, start_time)
    plot('TEMP', mintemp, maxtemp, subject_name, start_time)


def plotEEG(axisA, axisB, subject_name, start_time):
    df = pd.read_csv('data_processed/' + subject_name + '_EEG.csv', usecols=[axisA, axisB, 'TIME'])
    df[axisA + ' / ' + axisB] = df[axisA]/df[axisB]
    minhr = 10000000
    maxhr = -10000000
    for x in list(df[axisA + ' / ' + axisB]):
        minhr = min(minhr, x)
        maxhr = max(maxhr, x)
    fig = px.line(df, x='TIME', y=axisA + ' / ' + axisB)
    fig.add_shape(type="line",
        x0 = str(start_time.time()), y0 = minhr, x1 = str(start_time.time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_training)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_training)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_relax)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_relax)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_control)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_control)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_rest)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_rest)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_test)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_test)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_rest2)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_rest2)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_control2)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_control2)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.add_shape(type="line",
        x0 = str((start_time + datetime.timedelta(seconds=duration_complete)).time()), y0 = minhr,
        x1 = str((start_time + datetime.timedelta(seconds=duration_complete)).time()), y1 = maxhr,
        line = dict(color="Red", width=1, dash="dashdot")
    )
    fig.write_image("results/" + subject_name + "_" + axisA + "_by_" + axisB + ".png")

    line = 0
    X = list(df['TIME'])
    Y = list(df[axisA + ' / ' + axisB])
    point1 = (start_time).time()
    point2 = (start_time + datetime.timedelta(seconds=duration_training)).time()
    point3 = (start_time + datetime.timedelta(seconds=duration_relax)).time()
    point4 = (start_time + datetime.timedelta(seconds=duration_control)).time()
    point5 = (start_time + datetime.timedelta(seconds=duration_rest)).time()
    point6 = (start_time + datetime.timedelta(seconds=duration_test)).time()
    point7 = (start_time + datetime.timedelta(seconds=duration_rest2)).time()
    point8 = (start_time + datetime.timedelta(seconds=duration_control2)).time()
    point9 = (start_time + datetime.timedelta(seconds=duration_complete)).time()
    vals = [0, 0, 0, 0, 0, 0, 0, 0]
    cnt = [0, 0, 0, 0, 0, 0, 0, 0]
    while (line < len(X)):
        cur = datetime.datetime.strptime(X[line], '%H:%M:%S')
        if cur.time() >= point1 and cur.time() < point2:
            vals[0] = vals[0] + float(Y[line])
            cnt[0] = cnt[0] + 1
        if cur.time() >= point2 and cur.time() < point3:
            vals[1] = vals[1] + float(Y[line])
            cnt[1] = cnt[1] + 1
        if cur.time() >= point3 and cur.time() < point4:
            vals[2] = vals[2] + float(Y[line])
            cnt[2] = cnt[2] + 1
        if cur.time() >= point4 and cur.time() < point5:
            vals[3] = vals[3] + float(Y[line])
            cnt[3] = cnt[3] + 1
        if cur.time() >= point5 and cur.time() < point6:
            vals[4] = vals[4] + float(Y[line])
            cnt[4] = cnt[4] + 1
        if cur.time() >= point6 and cur.time() < point7:
            vals[5] = vals[5] + float(Y[line])
            cnt[5] = cnt[5] + 1
        if cur.time() >= point7 and cur.time() < point8:
            vals[6] = vals[6] + float(Y[line])
            cnt[6] = cnt[6] + 1
        if cur.time() >= point8 and cur.time() < point9:
            vals[7] = vals[7] + float(Y[line])
            cnt[7] = cnt[7] + 1
        line = line+1

    for i, val in enumerate(cnt):
        if val > 0:
            vals[i] = vals[i] / val
    legends = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9"]
    newdf = pd.DataFrame(list(zip(legends, vals)), columns=['legends', axisA + ' / ' + axisB])
    fig2 = px.line(newdf, x='legends', y=axisA + ' / ' + axisB)
    fig2.write_image("results/[AVG]" + subject_name + "_" + axisA + "_by_" + axisB + ".png")


def performEEG(location, subject_name, start_time):
    data = pd.read_csv(location + "/data.csv", sep=',', skiprows=1, error_bad_lines=False, index_col=False, dtype='unicode')
    columns = {}
    timeAdded = False
    for col in data.columns:
        if "POW." in col:
            correctedCol = []
            addi = []
            ind = 0
            for vals in list(data[col]):
                if vals is np.nan:
                    ind = ind+1
                    continue
                if timeAdded is False:
                    addi.append(str(time.strftime('%H:%M:%S', time.localtime(float(data["Timestamp"][ind])))))
                correctedCol.append(vals)
                ind = ind+1
            if timeAdded is False:
                columns["TIME"] = addi
                timeAdded = True
            columns[str(col)] = correctedCol
    
    rows = []
    line = 0
    while line < len(columns["TIME"]):
        cur = []
        for key in columns.keys():
            cur.append(columns[key][line])
        rows.append(cur)
        line = line+1

    with open('data_processed/' + subject_name + '_EEG.csv', 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(columns.keys())
        write.writerows(rows)
    
    for band in ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]:
        plotEEG("POW." + band + ".Alpha", "POW." + band + ".BetaL", subject_name, start_time)
        plotEEG("POW." + band + ".Alpha", "POW." + band + ".BetaH", subject_name, start_time)
        plotEEG("POW." + band + ".Theta", "POW." + band + ".BetaL", subject_name, start_time)
        plotEEG("POW." + band + ".Theta", "POW." + band + ".BetaH", subject_name, start_time)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'r':
            shutil.rmtree('data_processed')
            shutil.rmtree('results')
            os.mkdir("data_processed")
            os.mkdir("results")

    for subject in os.listdir('data'):
        infile = open('data/' + subject + '/Log.txt', 'r')
        start_time = datetime.datetime.strptime(infile.readline().replace('\n', ''), '%H:%M:%S')
        print('Performing on ' + subject + ' start_time=' + str(start_time.time()))
        perform('data/' + subject + '/WatchData', subject, start_time)
        print("ECG Done!")
        performEEG('data/' + subject, subject, start_time)
        print("EEG Done!")

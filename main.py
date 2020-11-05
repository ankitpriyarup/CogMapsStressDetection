import os
import sys
import csv
import time
import datetime
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go


pio.orca.config.executable = 'C:/Users/ankitpriyarup/AppData/Local/Programs/orca/orca.exe'
pio.orca.config.save()

def plot(axis, minhr, maxhr, subject_name, start_time):
    duration_training = 3*60 + 2
    duration_relax = 8*60 + 32
    duration_control = 17*60 + 6
    duration_rest = 18*60 + 5
    duration_test = 30*60 + 7
    duration_rest2 = 31*60 + 37
    duration_control2 = 40*60 + 9
    duration_complete = 45*60 + 40

    df = pd.read_csv('data_processed/' + subject_name + '.csv')
    
    fig = px.line(df, x='TIME', y=axis)
    fig.add_trace(go.Scatter(
        x=[str(start_time.time()),
            str((start_time + datetime.timedelta(seconds=duration_training)).time()),
            str((start_time + datetime.timedelta(seconds=duration_relax)).time()),
            str((start_time + datetime.timedelta(seconds=duration_control)).time()),
            str((start_time + datetime.timedelta(seconds=duration_rest)).time()),
            str((start_time + datetime.timedelta(seconds=duration_test)).time()),
            str((start_time + datetime.timedelta(seconds=duration_rest2)).time()),
            str((start_time + datetime.timedelta(seconds=duration_control2)).time()),
            str((start_time + datetime.timedelta(seconds=duration_complete)).time())],
        y=[minhr, minhr, minhr, minhr, minhr, minhr, minhr, minhr, minhr],
        text=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        mode="text",
    ))
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

    with open('data_processed/' + subject_name + '.csv', 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(fields)
        write.writerows(rows)

    plot('HR', minhr, maxhr, subject_name, start_time)
    plot('EDA', mineda, maxeda, subject_name, start_time)
    plot('TEMP', mintemp, maxtemp, subject_name, start_time)


if __name__ == "__main__":
    for subject in os.listdir('data'):
        infile = open('data/' + subject + '/Log.txt', 'r')
        start_time = datetime.datetime.strptime(infile.readline().replace('\n', ''), '%H:%M:%S')
        print('Performing on ' + subject + ' start_time=' + str(start_time.time()))
        perform('data/' + subject + '/WatchData', subject, start_time)

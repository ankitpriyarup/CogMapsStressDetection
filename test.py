import pandas as pd
import numpy as np
import os, sys, time, datetime, csv, pyeeg
from scipy.signal import butter, lfilter, freqz

BAND = [0.5, 4, 7, 12, 30]      # Delta, Theta, Alpha, and Beta
SAMPLING_FREQUENCY = 128
BAND_PASS_FILTER_ORDER = 4
LOW_BAND_PASS_RANGE = 0.5
HIGH_BAND_PASS_RANGE = 30
FEATURE_COMBINE_ORDER = 16
DURATIONS = [3*60 + 2, 8*60 + 32, 17*60 + 6, 18*60 + 5, 30*60 + 7, 31*60 + 37, 40*60 + 9, 45*60 + 40]
# CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
CHANNELS = ["AF3"]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def calculate_features(signal):
    # signal_filtered = butter_bandpass_filter(signal, LOW_BAND_PASS_RANGE, HIGH_BAND_PASS_RANGE, SAMPLING_FREQUENCY, BAND_PASS_FILTER_ORDER)
    signal_filtered = signal

    hjorth = pyeeg.hjorth(signal_filtered, D=None)
    binary_power = pyeeg.bin_power(signal_filtered, BAND, SAMPLING_FREQUENCY)
    spectral_entropy = pyeeg.spectral_entropy(signal_filtered, BAND, SAMPLING_FREQUENCY, binary_power)

    return { 'MOBILITY':hjorth[0], 'COMPLEXITY':hjorth[1], 'POW_THETA':binary_power[0][1], 'POW_ALPHA':binary_power[0][2], 'POW_BETA':binary_power[0][3],
             'POW_RATIO_THETA':binary_power[1][1], 'POW_RATIO_ALPHA':binary_power[1][2], 'POW_RATIO_BETA':binary_power[1][3], 'SPECTRAL_ENTROPY_THETA':spectral_entropy[1],
             'SPECTRAL_ENTROPY_ALPHA':spectral_entropy[2], 'SPECTRAL_ENTROPY_BETA':spectral_entropy[3] }


def performEEG(location, subject_name, start_time):
    data = pd.read_csv(location + "/data.csv", sep=',', skiprows=1, error_bad_lines=False, index_col=False, dtype='unicode')
    points = [(start_time).time(), (start_time + datetime.timedelta(seconds=DURATIONS[0])).time(), (start_time + datetime.timedelta(seconds=DURATIONS[1])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[2])).time(), (start_time + datetime.timedelta(seconds=DURATIONS[3])).time(), (start_time + datetime.timedelta(seconds=DURATIONS[4])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[5])).time(), (start_time + datetime.timedelta(seconds=DURATIONS[6])).time(), (start_time + datetime.timedelta(seconds=DURATIONS[7])).time()]
    res = {}
    res['PHASE'], res['TIME'], rows, cum_signal, pos, counter = [], [], [], {}, 0, 0
    for i in range(0, len(data)):
        curTimeStr = str(time.strftime('%H:%M:%S', time.localtime(float(data["Timestamp"][i]))))
        curTime = datetime.datetime.strptime(curTimeStr, '%H:%M:%S')
        if (curTime.time() >= points[0] or pos > 0) and pos < 9:
            if pos == 0 or curTime.time() == points[pos]:
                pos = pos+1

            if counter%FEATURE_COMBINE_ORDER == 0 and counter > 0:
                res['TIME'].append(data["Timestamp"][i])
                res['PHASE'].append(str(pos) + '-' + str(pos+1))
                for channel in CHANNELS:
                    for key, value in calculate_features([float(i) for i in np.array(cum_signal[channel])]).items():
                        if channel + '_' + key not in res:
                            res[channel + '_' + key] = []
                        res[channel + '_' + key].append(value)
                cum_signal = {}

            for channel in CHANNELS:
                if channel not in cum_signal:
                    cum_signal[channel] = []
                cum_signal[channel].append(data['EEG.' + channel][i])
            counter = counter+1
            if counter > 1000:
                break

    labels, rows = [], []
    for i in range(0, len(res['TIME'])):
        cur = []
        fillLabel = len(labels) == 0
        for key, value in res.items():
            if fillLabel:
                labels.append(key)
            cur.append(value[i])
        rows.append(cur)
    with open('data_processed/' + subject_name + '.csv', 'w', newline='') as file:
        write = csv.writer(file)
        write.writerow(labels)
        write.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'r':
        shutil.rmtree('data_processed')
        shutil.rmtree('results')
        os.mkdir("data_processed")
        os.mkdir("results")

    for subject in os.listdir('data'):
        infile = open('data/' + subject + '/Log.txt', 'r')
        start_time = datetime.datetime.strptime(infile.readline().replace('\n', ''), '%H:%M:%S')
        print('Performing on ' + subject + ' start_time=' + str(start_time.time()))
        performEEG('data/' + subject, subject, start_time)
        break

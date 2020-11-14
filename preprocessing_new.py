import pandas as pd
import numpy as np
import os, sys, time, datetime, csv, pyeeg
from scipy.signal import butter, lfilter, freqz

BAND = [0.5, 4, 7, 12, 30]      # Delta, Theta, Alpha, and Beta
SAMPLING_FREQUENCY = 128
FEATURE_COMBINE_ORDER = SAMPLING_FREQUENCY // 4     # 0.25 SECOND
DURATIONS = [3*60 + 2, 8*60 + 32, 17*60 + 6, 18*60 + 5, 30*60 + 7, 31*60 + 37, 40*60 + 9, 45*60 + 40]
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpower(data, sf, band, window_sec=None, relative=False):
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf
    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = simps(psd[idx_band], dx=freq_res)
    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def calculate_eeg_features(signal):
    minimum = np.amin(signal)
    maximum = np.amax(signal)
    mean = np.mean(signal)
    standard_deviation = np.std(signal)
    hjorth = pyeeg.hjorth(signal, D=None)
    theta = bandpower(signal, SAMPLING_FREQUENCY, [4, 8], FEATURE_COMBINE_ORDER/SAMPLING_FREQUENCY)
    alpha = bandpower(signal, SAMPLING_FREQUENCY, [8, 12], FEATURE_COMBINE_ORDER/SAMPLING_FREQUENCY)
    beta = bandpower(signal, SAMPLING_FREQUENCY, [12, 30], FEATURE_COMBINE_ORDER/SAMPLING_FREQUENCY)
    total = bandpower(signal, SAMPLING_FREQUENCY, [4, 100], FEATURE_COMBINE_ORDER/SAMPLING_FREQUENCY)
    if theta == 0 or alpha == 0 or beta == 0:
        print(signal)
    # spectral_entropy = pyeeg.spectral_entropy(signal, BAND, SAMPLING_FREQUENCY, pyeeg.bin_power(signal, BAND, SAMPLING_FREQUENCY)[1])

    return { 'MINIMUM':minimum, 'MAXIMUM':maximum, 'MEAN':mean, 'STANDARD_DEVIATION':standard_deviation, 'MOBILITY':hjorth[0], 'COMPLEXITY':hjorth[1],
             'POW_ALPHA_BY_BETA':abs(alpha)/abs(beta), 'POW_THETA_BY_ALPHA':abs(theta)/abs(alpha), 'POW_THETA_RELATIVE':abs(theta)/abs(total),
             'POW_ALPHA_RELATIVE':abs(alpha)/abs(total), 'POW_BETA_RELATIVE':abs(beta)/abs(total), 'SPECTRAL_ENTROPY':0 }


def calculate_ecg_features(eda, hr, temp):
    mean_eda = np.mean(eda)
    mean_hr = np.mean(hr)
    mean_temp = np.mean(temp)

    return { 'EDA':mean_eda, 'HR':mean_hr, 'TEMP':mean_temp }


def performPreprocessing(location, subject_name, start_time):
    data_eeg = pd.read_csv(location + "/data.csv", sep=',', skiprows=1, error_bad_lines=False, index_col=False, dtype='unicode')
    data_eda, data_hr, data_temp = pd.read_csv(location + "/WatchData/EDA.csv"), pd.read_csv(location + "/WatchData/HR.csv"), pd.read_csv(location + "/WatchData/TEMP.csv")
    points = [(start_time).time(), (start_time + datetime.timedelta(seconds=DURATIONS[0])).time(), (start_time + datetime.timedelta(seconds=DURATIONS[1])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[2])).time(), (start_time + datetime.timedelta(seconds=DURATIONS[3])).time(), (start_time + datetime.timedelta(seconds=DURATIONS[4])).time(),
              (start_time + datetime.timedelta(seconds=DURATIONS[5])).time(), (start_time + datetime.timedelta(seconds=DURATIONS[6])).time(), (start_time + datetime.timedelta(seconds=DURATIONS[7])).time()]
    
    # 4, 1, 4 Hz freq respectively, upsampling it uniformly at 128 Hz (to match with EEG)
    # HR starts at 10 sec delay, so offset others accordingly
    eda, hr, temp = [], [], []
    for val in data_eda.values.tolist()[41:]:
        for i in range(0, SAMPLING_FREQUENCY//4):
            eda.append(val)
    for val in data_hr.values.tolist()[1:]:
        for i in range(0, SAMPLING_FREQUENCY):
            hr.append(val)
    for val in data_temp.values.tolist()[41:]:
        for i in range(0, SAMPLING_FREQUENCY//4):
            temp.append(val)
    ecg_cur_time = curTime = datetime.datetime.strptime(str(time.strftime('%H:%M:%S', time.localtime(float(data_hr.columns[0])))), '%H:%M:%S')
    offset = 0
    for i in range(0, len(eda)):
        if (i%SAMPLING_FREQUENCY == 0 and i > 0):
            ecg_cur_time = ecg_cur_time + datetime.timedelta(seconds=1)
        if ecg_cur_time.time() >= points[0]:
            offset = i
            break

    res = {}
    res['PHASE'], res['TIME'], rows, cum_eeg_signal, pos, counter = [], [], [], {}, 0, 0
    for i in range(0, len(data_eeg)):
        curTime = datetime.datetime.strptime(str(time.strftime('%H:%M:%S', time.localtime(float(data_eeg["Timestamp"][i])))), '%H:%M:%S')
        if (curTime.time() >= points[0] or pos > 0) and pos < 9:
            if pos == 0 or curTime.time() == points[pos]:
                pos = pos+1

            if counter%FEATURE_COMBINE_ORDER == 0 and counter > 0:
                res['TIME'].append(data_eeg["Timestamp"][i])
                res['PHASE'].append(str(pos) + ' -> ' + str(pos+1))

                # ECG
                l, r = offset+counter-FEATURE_COMBINE_ORDER, offset+counter
                for key, value in calculate_ecg_features(eda[l:r], hr[l:r], temp[l:r]).items():
                    if key not in res:
                        res[key] = []
                    res[key].append(value)

                # EEG
                for channel in CHANNELS:
                    for key, value in calculate_eeg_features([float(i) for i in np.array(cum_eeg_signal[channel])]).items():
                        if channel + '_' + key not in res:
                            res[channel + '_' + key] = []
                        res[channel + '_' + key].append(value)
                cum_eeg_signal = {}

            for channel in CHANNELS:
                if channel not in cum_eeg_signal:
                    cum_eeg_signal[channel] = []
                cum_eeg_signal[channel].append(data_eeg['EEG.' + channel][i])
            counter = counter+1

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
        if "13" not in subject:
            continue
        infile = open('data/' + subject + '/Log.txt', 'r')
        start_time = datetime.datetime.strptime(infile.readline().replace('\n', ''), '%H:%M:%S')
        print('Performing on ' + subject + ' start_time=' + str(start_time.time()))
        performPreprocessing('data/' + subject, subject, start_time)

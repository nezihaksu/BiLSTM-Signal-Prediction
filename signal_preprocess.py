from scipy.signal import butter, lfilter
import math
import numpy
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import itertools
import time

from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

import os
import xlrd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

raw_value = pd.read_csv(r"C:\Users\nezih\Desktop\hw3\sensor1104-6e22.csv")
targets = pd.read_excel(r"C:\Users\nezih\Desktop\hw3\label1104-6e22.xlsx")

print(raw_value)
print(targets)

train_start_hour = "13"
train_start_minute = "15"
train_end_hour = "13"
train_end_minute = "30"

test_start_hour = "13"
test_start_minute = "31"
test_end_hour = "13"
test_end_minute = "43"




class signal_processing:
    def __init__(self, value, id_number):
        self.value = value
        self.id_number = id_number

    def butter_bandpass(self, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def run_butter_filter(self, x, lowcut, highcut, fs, order):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = self.butter_bandpass_filter(x, lowcut, highcut, fs, order)
        return y

    def moving_avarage_filter(self, x, order):
        b = (1 / order) * np.ones((order, 1))
        y = lfilter(b.flatten(), 1.0, x.flatten())
        return y

    def segmentation_fn(self, temp, seg_sample_num, seg_nember, bin_length):
        y = []
        for x in range(seg_nember):
            thr = np.var(temp[x, :], ddof=1) * 0.09
            indices = [1, seg_sample_num - 2]
            a = 0
            b = 1
            while indices[a] != seg_sample_num - 2:
                if np.var(temp[x, indices[a]:indices[b] + 1], ddof=1) > thr:
                    c = math.floor((indices[b] + indices[a]) / 2)
                    if c in indices:
                        a = a + 1
                        b = b + 1
                    else:
                        indices.append(c)
                        indices.sort()
                else:
                    a = a + 1
                    b = b + 1

            if len(indices) > bin_length:
                var_bin = []
                for j in range(len(indices) - 1):
                    var_bin.append([np.var(temp[x, indices[j]:indices[j + 1] + 1], ddof=1), j])
                var_bin.sort()
                sort_index = []
                for x in var_bin:
                    sort_index.append(x[1])
                index_flip = np.flip(sort_index)
                new_idx = index_flip[0:bin_length - 1]
                new_idx1 = []
                for w in range(len(new_idx)):
                    new_idx1.append(indices[new_idx[w]])
                new_idx1.sort()
                y.append(new_idx1)
            else:
                l = math.floor(seg_sample_num / (bin_length - 1))
                d = []
                for i in range(bin_length - 1):
                    d.append(i * l + 1)
                y.append(d)
        return y

    def signalProcessingFun(self):
        ##### butherworth bandpass filtering###

        fs = 100.0
        lowcut = 1.0
        highcut = 9.0
        order = 5
        dataOut_filter = self.run_butter_filter(self.value, lowcut, highcut, fs, order)

        ##### Reduction Of abnormality in signal by superposition of moving averages and normalization

        data_length = len(dataOut_filter)
        segmentation_length = 100
        filter_order = 8
        segments_numbers = math.floor(data_length / segmentation_length)
        temp = np.subtract(dataOut_filter, np.mean(dataOut_filter))
        temp1 = np.reshape(temp[0:segments_numbers * segmentation_length], (segments_numbers, segmentation_length))

        out_moving_filter = np.zeros((segments_numbers, segmentation_length))
        temp11 = np.zeros((segments_numbers, segmentation_length))
        for i in range(1, segments_numbers):
            out_moving_filter[i, :] = self.moving_avarage_filter(temp1[i, :], filter_order)

        var_segmentation = []
        for element in out_moving_filter:
            var_segmentation.append(np.var(element, ddof=1) / 10)

        for i in range(1, segments_numbers):
            temp11[i, :] = temp1[i, :] / var_segmentation[i]

        dataOut = temp11.flatten()
        dataOut_filter = self.run_butter_filter(dataOut, lowcut, highcut, fs, order)
        dataOut = np.subtract(dataOut_filter, np.mean(dataOut_filter))
        dataOut = np.true_divide(dataOut, np.sqrt(np.var(dataOut, ddof=1)))

        ##### adaptive segmentations

        bins_num = self.id_number
        bin_length = 4
        seg_sample_num = math.floor(len(dataOut) / bins_num)
        temp = np.reshape(dataOut[0:bins_num * seg_sample_num], (bins_num, seg_sample_num))
        bin_arrange = []
        bin_arrange = self.segmentation_fn(temp, seg_sample_num, bins_num, bin_length)

        id_list = []
        id_samples = []
        num = 0
        for x in range(bins_num):
            temp1 = bin_arrange[x]
            temp11 = temp[x, :]
            r = 0
            for y in range(bin_length):
                if y == bin_length - 1:
                    id_list.append([num] * (seg_sample_num - r))
                    id_samples.append((temp11[r:seg_sample_num]))
                else:
                    id_list.append([num] * (temp1[y] - r))
                    id_samples.append((temp11[r:temp1[y]]))
                    r = temp1[y]
                num = num + 1

        return bins_num, id_list, id_samples, seg_sample_num, bin_length


def preprocess(raw_scg, start_hour, start_min, end_hour, end_min):
    start_time = "{}:{}".format(start_hour, start_min)
    end_time = "{}:{}".format(end_hour, end_min)

    print(start_time)
    print(end_min)

    targets_start_index = targets[targets["Time"].astype(str).str.contains(start_time)].iloc[0, :].name
    targets_end_index = targets[targets["Time"].astype(str).str.contains(end_time)].iloc[0, :].name

    id_number = targets_end_index - targets_start_index
    raw_value_start_index = raw_value[raw_value.Time.str.contains(start_time)].iloc[0, :].name
    raw_value_end_index = raw_value[raw_value.Time.str.contains(end_time)].iloc[0, :].name - 1
    raw_scg_value = raw_scg.iloc[raw_value_start_index:raw_value_end_index, 1].values

    preprocessed_scg = signal_processing(raw_scg_value, id_number)
    new_id_number, id_list, id_samples, sample_size, bin_length = preprocessed_scg.signalProcessingFun()

    id_list_flatten = list(itertools.chain(*id_list))
    id_samples_flatten = list(itertools.chain(*id_samples))

    df = pd.DataFrame(id_samples_flatten)
    df["id"] = id_list_flatten
    respected_time = raw_value.Time[0:len(id_samples_flatten)]
    new_time = []
    seg = new_id_number
    for x in range(seg):
        for y in range(sample_size):
            new_time.append(raw_value.Time[x * sample_size + y])

    df['time'] = np.array(new_time)

    return new_id_number, bin_length,targets_start_index, df.dropna(axis=0)


def data_segmenation_normalization(data):
    df = data
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n * 0.7)]  ##70% training Data
    val_df = df[int(n * 0.7):int(n * 0.9)]  # 20% test data
    test_df = df[int(n * 0.9):]  # 10% predicting data

    num_features = df.shape[1]
    train_mean = train_df.mean()
    train_std = train_df.std()
    # Normalization
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    return train_df, val_df, test_df


def data_segmenation_normalization1(data):
    df = data
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n * 0.7)]  ##70% training Data

    num_features = df.shape[1]
    train_mean = train_df.mean()
    train_std = train_df.std()
    # Normalization
    new_data = (data - train_mean) / train_std

    return new_data


def data_shaping(data, S, D, H, R):
    data["S"] = np.array(S)
    data["D"] = np.array(D)
    data["H"] = np.array(H)
    data["R"] = np.array(R)

    df = data
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n * 0.7)]  ##70% training Data
    val_df = df[int(n * 0.7):int(n * 0.9)]  # 20% test data
    test_df = df[int(n * 0.9):]  # 10% predicting data

    return train_df, val_df, test_df


features = ['0__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.6',
            '0__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.6',
            '0__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6',
            '0__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6',
            '0__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6',
            '0__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
            '0__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
            '0__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
            '0__quantile__q_0.1', '0__quantile__q_0.2',
            '0__quantile__q_0.3', '0__quantile__q_0.4', '0__quantile__q_0.6', '0__quantile__q_0.7',
            '0__quantile__q_0.8', '0__quantile__q_0.9', '0__percentage_of_reoccurring_values_to_all_values',
            '0__percentage_of_reoccurring_datapoints_to_all_datapoints', '0__sum_of_reoccurring_values',
            '0__sum_of_reoccurring_data_points', '0__ratio_value_number_to_time_series_length',
            '0__time_reversal_asymmetry_statistic__lag_2', '0__time_reversal_asymmetry_statistic__lag_3',
            '0__c3__lag_1', '0__c3__lag_2', '0__c3__lag_3', '0__maximum', '0__minimum']


def label_modification_df(targets, targets_start_index, targets_end_index, bin_number):
    targets = targets[targets_start_index:targets_start_index+targets_end_index].reset_index()
    S = []
    D = []
    H = []
    R = []
    df = pd.DataFrame()
    df['S'] = targets.S
    df['D'] = targets.D
    df['H'] = targets.H
    df['R'] = targets.R

    coef = 1
    for x in targets.S:
        for y in range(bin_number):
            S.append(x/coef)

    for x in targets.D:
        for y in range(bin_number):
            D.append(x/coef)

    for x in targets.H:
        for y in range(bin_number):
            H.append(x/coef)

    for x in targets.R:
        for y in range(bin_number):
            R.append(x/coef)

    return S, D, H, R,df


def data_normalization(data):
    df = data
    df_mean = df.mean()
    df_std = df.std()

    # Normalization
    df = (df - df_mean) / df_std

    return df


# Processing first 20 min of raw scg signal and creating dataset with adaptive indexes and corresponding time row to extract features according to them.
# Arguments are in order:start_hour,start_min,end_hour,end_min
new_id_number_train, bin_length_train,targets_startInx_train, train_df = preprocess(raw_value, train_start_hour, train_start_minute,
                                                             train_end_hour, train_end_minute)

# Processing last 7 minutes of raw scg signal to predict future parameters.
new_id_number_test, bin_length_test,targets_startInx_test, test_df = preprocess(raw_value, test_start_hour, test_start_minute, test_end_hour,
                                                          test_end_minute)

# Because of problems occuring while parallel processing i had to set n_jobs = 0,it calculates slower but works fine.
train_extracted_features = extract_features(train_df, column_id="id", column_sort="time",
                                           default_fc_parameters=EfficientFCParameters(), n_jobs=0)
test_extracted_features = extract_features(test_df, column_id="id", column_sort="time",
                                          default_fc_parameters=EfficientFCParameters(), n_jobs=0)

# train_extracted_features.to_csv('train_final_features.csv',index = False)
#tt = pd.read_csv(r'C:\Users\Samane\Desktop\hw\20minFeature.xlsx')
# test_extracted_features.to_csv('test_final_features.csv',index = False)

train_features = train_extracted_features[features]
train_features_norm = data_segmenation_normalization1(train_features)
S, D, H, R, labels_train= label_modification_df(targets, targets_startInx_train, new_id_number_train, bin_length_train)

train_df, val_df, test_df = data_shaping(train_features_norm, S, D, H, R)

test_array = test_df.values
test_array = np.expand_dims(test_array, axis=0)

#test_features = pd.read_csv(r'C:\Users\Samane\Desktop\hw\7mintest.xlsx')[features]
test_features = test_extracted_features[features]
test_features_norm = data_normalization(test_features)
S, D, H, R, labels_test= label_modification_df(targets, targets_startInx_test, new_id_number_test, bin_length_test)

test_features_norm['S'] = np.array(S)
test_features_norm["D"] = np.array(D)
test_features_norm["H"] = np.array(H)
test_features_norm["R"] = np.array(R)

test_features_norm = test_features_norm.values
test_features_norm = np.expand_dims(test_features_norm, axis=0)
print("TEST FEATURE NORM SHAPE")
print(test_features_norm.shape)


# Data windowing for BiLSTM model
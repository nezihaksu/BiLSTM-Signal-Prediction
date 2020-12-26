from scipy.signal import butter, lfilter
import math
import numpy as np
import pandas as pd
import itertools

raw_value = pd.read_csv(r"C:\Users\nezih\Desktop\hw3\sensor-27minutes.csv")
targets = pd.read_excel(r"C:\Users\nezih\Desktop\hw3\label-27minutes.xlsx")
wb = xlrd.open_workbook(r"C:\Users\nezih\Desktop\hw3\label-27minutes.xlsx")
sheet = wb.sheet_by_index(0)

train_start_hour = 12
train_start_minute = 19
train_end_hour = 12
train_end_minute = 39

test_start_hour = 12
test_start_minute = 39
test_end_hour = 12
test_end_minute = 46


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
            thr = np.var(temp[x, :], ddof=1) * 0.7
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
                l = math.floor(seg_sample_num / 15)
                d = []
                for i in range(15):
                    d.append([i * l])
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
        bin_length = 16
        seg_nember = math.floor(bins_num / bin_length)
        seg_sample_num = math.floor(len(dataOut) / seg_nember)
        temp = np.reshape(dataOut[0:seg_nember * seg_sample_num], (seg_nember, seg_sample_num))
        bin_arrange = []
        bin_arrange = self.segmentation_fn(temp, seg_sample_num, seg_nember, bin_length)

        id_list = []
        id_samples = []
        num = 0
        for x in range(seg_nember):
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

        return seg_nember*bin_length,id_list, id_samples

def label_filtering():
    y = []
    # for i in range(sheet.nrows):
    #     if sheet.cell_value(i, 5) == 0 or sheet.cell_value(i, 5) == nan:
    #         y.append(i)
    return y

def preprocess(raw_scg, start_hour, start_min, end_hour, end_min):

    start_time = "{}:{}".format(start_hour, start_min)
    end_time = "{}:{}".format(end_hour, end_min)

    targets_start_index = targets[targets["Raw Time"].astype(str).str.contains(start_time)].iloc[0, :].name
    targets_end_index = targets[targets["Raw Time"].astype(str).str.contains(end_time)].iloc[0, :].name

    id_number = targets_end_index - targets_start_index
    raw_value_start_index = raw_value[raw_value.Time.str.contains(start_time)].iloc[0, :].name
    raw_value_end_index = raw_value[raw_value.Time.str.contains(end_time)].iloc[0, :].name - 1
    raw_scg_value = raw_scg.iloc[raw_value_start_index:raw_value_end_index, 1].values

    preprocessed_scg = signal_processing(raw_scg_value, id_number)
    new_id_number, id_list, id_samples = preprocessed_scg.signalProcessingFun()

    id_list_flatten = list(itertools.chain(*id_list))
    distinct_id_list = set(id_list_flatten)
    print("NUMBER OF DISTINCT IDS")
    print(len(distinct_id_list))
    id_samples_flatten = list(itertools.chain(*id_samples))

    df = pd.DataFrame(id_samples_flatten)
    df["id"] = id_list_flatten
    respected_time = raw_value.Time[0:len(id_samples_flatten)]
    df['time'] = respected_time

    return new_id_number,df.dropna(axis=0)
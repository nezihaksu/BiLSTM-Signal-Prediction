import numpy
from scipy.signal import butter, lfilter
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

label = pd.read_csv(r"C:\Users\nezih\Desktop\hw3\sensor-27minutes.csv")
data_input = label.iloc[:, 1].values


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run_butter_filter(x, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order)
    return y


def moving_avarage_filter(x, order):
    b = (1 / order) * np.ones((order, 1))
    y = lfilter(b.flatten(), 1.0, x.flatten())
    return y

def segmentation_fn(temp,seg_sample_num,seg_nember):
    y = []
    for x in range(seg_nember):
        thr = np.var(temp[x, :], ddof=1) * 0.8
        indices = [0, seg_sample_num - 2]
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
            index_flip = numpy.flip(sort_index)
            new_idx = index_flip[0:bin_length]
            new_idx1 = []
            for w in range(len(new_idx)):
                new_idx1.append(indices[new_idx[w]])
            new_idx1.sort()
            y.append(new_idx1)
    return y

            ##### butherworth bandpass filtering###

fs = 100.0
lowcut = 1.0
highcut = 8.0
order = 4
dataOut_filter = run_butter_filter(data_input, lowcut, highcut, fs, order)

##### Reduction Of abnormality in signal by superposition of moving averages and normalization

data_length = len(dataOut_filter)
filter_order = 8
out_moving_filter = moving_avarage_filter(dataOut_filter, filter_order)
out_mf1 = dataOut_filter[0:len(dataOut_filter)-3] - out_moving_filter[3:len(dataOut_filter)]

out_mf11 = np.subtract(out_mf1,np.mean(out_mf1))
out_mf111 = np.true_divide(out_mf11,np.sqrt(np.var(out_mf11,ddof=1)))

# plt.plot(out_mf111)
# plt.show()

##### adaptive segmentations

bins_num = 1200#len()
bin_length = 16
seg_nember = math.floor(bins_num / bin_length)
seg_sample_num = math.floor(len(out_mf111)/seg_nember)
temp = np.reshape(out_mf111[0:seg_nember*seg_sample_num],(seg_nember,seg_sample_num))
bin_arrange = []
bin_arrange = segmentation_fn(temp,seg_sample_num,seg_nember)

import itertools

id_list = []
id_samples = []
num = 0
for x in range(seg_nember):
    temp1 = bin_arrange[x]
    temp11 = temp[x,:]
    r = 0
    for y in range(bin_length):
        id_list.append([num] * (temp1[y] - r))
        id_samples.append((temp11[r:temp1[y]]))
        r= temp1[y]
        num = num + 1


id_list_flatten = list(itertools.chain(*id_list))
id_samples_flatten = list(itertools.chain(*id_samples))

df = pd.DataFrame(id_samples_flatten)
df["id"] = id_list_flatten
print(df)



#### deep learning Algorithm
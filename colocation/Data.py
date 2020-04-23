import os
import numpy as np
import sys
import torch.utils.data
import torch
from torch.utils.data import DataLoader, TensorDataset
import random


File_names = ['co2.csv', 'humidity.csv', 'light.csv', 'temperature.csv']

def clean_coequipment(ts, val, maxl = 30000):
    new_ts = [ts[0]]
    new_val = [val[0]]
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] == 1500:
            new_val.append(val[i])
        else:
            k = int((ts[i] - ts[i - 1]) / 1500)
            for _ in range(k):
                new_val.append(val[i])
    return new_val[0:maxl]

def read_ahu_csv(path, column = ['PropertyTimestampInNumber', 'SupplyFanSpeedOutput']):
    df = pd.read_csv(path)
    ts = df[column[0]]
    val = df[column[1]]
    return clean_coequipment(ts, val)

def read_vav_csv(path, column = ['PropertyTimestampInNumber', 'AirFlowNormalized']):
    df = pd.read_csv(path)
    ts = df[column[0]]
    val = df[column[1]]
    return clean_coequipment(ts, val)

def read_facility_ahu(facility_id, ahu_list):
    ahu_data, label = [], []
    path = "/localtmp/sl6yu/split/ahu_property_file_" + str(facility_id) + "/"
    for name in ahu_list[facility_id]:
        if os.path.exists(path + name + '.csv') == False:
            continue
        label.append(name)
        ahu_data.append(read_ahu_csv(path + name + '.csv'))
    return ahu_data, label

def read_facility_vav(facility_id, mapping):
    vav_data, label = [], []
    path = "/localtmp/sl6yu/split/vav_box_property_file_" + str(facility_id) + "/"
    for name in mapping[facility_id].keys():
        if os.path.exists(path + name + '.csv') == False:
            continue
        label.append(name)
        vav_data.append(read_vav_csv(path +name + '.csv'))
    return vav_data, label

def read_ground_truth(path = './mapping_data.xlsx'):
    data = pd.read_excel(path, sheet_name = 'Hierarchy Data', usecols=[1, 6, 7, 9])
    raw_list = data.values.tolist()
    mapping = dict()
    ahu_vas = dict()
    ahu_list = dict()
    for line in raw_list:
        if line[3] != 'AHU':
            continue
        f_id = int(line[0])
        parent_name = line[1]
        child_name = line[2]
        if f_id not in ahu_vas.keys():
            ahu_vas[f_id] = dict()
            ahu_list[f_id] = []
        ahu_list[f_id].append(child_name)
        ahu_vas[f_id][parent_name] = child_name

    for line in raw_list:
        if line[3] != 'VAV-BOX':
            continue
        f_id = int(line[0])
        parent_name = line[1]
        child_name = line[2]
        if f_id not in mapping.keys():
            mapping[f_id] = dict()
        if parent_name in ahu_vas[f_id].keys():
            mapping[f_id][child_name] = ahu_vas[f_id][parent_name]
    return mapping, ahu_list

    
def clean_temperature(value):
    for i in range(len(value)):
        if value[i] > 40 or value[i] < 10:
            if i == 0:
                value[i] = value[i + 1]
            else:
                value[i] = value[i - 1]
    return value

def read_colocation_data(config):
    folders = os.walk(config.data)
    x = []
    y = []
    true_pos = []
    cnt = 0
    for path, dir_list, _ in folders:  
        for dir_name in dir_list:
            folder_path = os.path.join(path, dir_name)
            for file in File_names:
                _, value = read_csv(os.path.join(folder_path, file), config)
                if file == 'temperature.csv':
                    value = clean_temperature(value)

                x.append(value)
                y.append(cnt)
                true_pos.append(dir_name)
            cnt += 1
    return x, y, true_pos


def align_length(ts, val, maxl, sample_f = 5):
    if len(val) >= maxl:
        return ts[0:maxl], val[0:maxl]
    else:
        for i in range(len(val), maxl):
            val.append(0) 
            ts.append(ts[-1] + sample_f)
        return ts, val

def read_csv(path, config):
    f = open(path)
    timestamps, vals = [], []
    for line in f.readlines():
        t, v = line.split(",")
        timestamps.append(int(t))
        vals.append(float(v))
    return align_length(timestamps, vals, config.max_length)

def sub_sample(ts, val, config):
    sample_f = config.interval
    MAXL = config.max_length

    min_ts = ts[0]
    max_ts = ts[-1]
    new_ts, new_val = [], []
    idx = 0
    for t in range(min_ts, max_ts - sample_f, sample_f):
        new_ts.append(t)
        tmp, cnt = 0, 0
        while ts[idx] < t + sample_f:
            tmp += val[idx]
            idx += 1
            cnt += 1
        if tmp != 0:
            new_val.append(tmp / cnt)
        else:
            new_val.append(tmp)
    return align_length(new_ts, new_val, MAXL, sample_f)

def STFT(x, config):
    fft_x = []
    for i in range(len(x)):
        fft_x.append(fft(x[i], config))
    return fft_x

def cross_validation_sample(total_cnt, test_cnt):
    assert total_cnt % test_cnt == 0

    folds = int(total_cnt / test_cnt)
    idx = list(range(total_cnt))
    random.shuffle(idx)
    test_index = []
    for i in range(folds):
        fold_index = []
        for j in range(test_cnt):
            fold_index.append(idx[test_cnt * i + j])
        test_index.append(fold_index)
    return test_index

def fft(v, config):

    stride = config.stride
    window_size = config.window_size
    k_coefficient = config.k_coefficient
    fft_data = []
    fft_freq = []
    power_spec =[]
    for i in range(int(len(v) / stride)):
        if stride * i + window_size > len(v):
            break
        v0 = v[stride * i: stride * i + window_size]
        v0 = np.array(v0)

        fft_window = np.fft.fft(v0)[1:k_coefficient+1]
        fft_flatten = np.array([fft_window.real, fft_window.imag]).astype(np.float32).flatten('F')
        fft_data.append(fft_flatten)

    return np.transpose(np.array(fft_data))

def split_colocation_train(x, y, test_index, split_method):

    train_x, train_y, test_x, test_y = [], [], [], []
    if split_method == 'room':
        for i in range(len(y)):
            if y[i] in test_index:
                test_x.append(x[i])
                test_y.append(y[i])
            else:
                train_x.append(x[i])
                train_y.append(y[i])
    else:
        for i in range(len(y)):
            if i not in test_index:
                train_x.append(x[i])
                train_y.append(y[i])
            else:
                test_y.append(i)
        test_x = x
    return train_x, train_y, test_x, test_y


def gen_colocation_triplet(train_x, train_y, prevent_same_type = False):

    triplet = []
    for i in range(len(train_x)): #anchor
        for j in range(len(train_x)): #negative
            if prevent_same_type and i % 4 == j % 4: 
                continue
            for k in range(len(train_x)): #positive
                if train_y[i] == train_y[j] or train_y[i] != train_y[k]:
                    continue
                if i == k:
                    continue
                sample = []
                sample.append(train_x[i])
                sample.append(train_x[k])
                sample.append(train_x[j])
                triplet.append(sample)
    return triplet

def gen_coequipment_triplet(ahu_x, train_vav, ahu_y, train_y, mapping):

    triplet = []

    for i in range(len(train_vav)): #anchor
        k = ahu_y.index(mapping[train_y[i]]) #postive
        for j in range(len(ahu_x)): #negative
            if j == k:
                continue
            sample = []
            sample.append(train_vav[i])
            sample.append(ahu_x[k])
            sample.append(ahu_x[j])
            triplet.append(sample)
    ''' 
    for i in range(len(ahu_x)): #anchor
        for j in range(len(train_vav)): #postive
            for k in range(len(train_vav)): #negative
                if mapping[train_y[j]] != ahu_y[i] or mapping[train_y[k]] == ahu_y[i]:
                    continue
                sample = []
                sample.append(ahu_x[i])
                sample.append(train_vav[k])
                sample.append(train_vav[j])
                triplet.append(sample)
    '''
    return triplet

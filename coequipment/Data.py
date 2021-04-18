import os
import numpy as np
import sys
import torch.utils.data
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import pandas as pd
import time

File_names = ['co2.csv', 'humidity.csv', 'light.csv', 'temperature.csv']

def clean_coequipment(ts, val):
    #timeArray = time.strptime(ts[0], "%m/%d/%Y %H:%M:%S")
    #new_ts = [int(time.mktime(timeArray))]
    new_val = []
    cnt = 0
    difs = []
    flag = False
    for i in range(len(val)):
        #timeArray = time.strptime(ts[i], "%m/%d/%Y %H:%M:%S")
        #new_ts.append(int(time.mktime(timeArray)))
        if np.isnan(val[i]):
            new_val.append(0)
        else:
            new_val.append(float(val[i]))
        if len(difs) > 5:
            flag = True
        elif new_val[-1] not in difs:
            difs.append(new_val[-1])
    return new_val, flag

def read_ahu_csv(path, column = ['PropertyTimestamp', 'SupplyFanSpeedOutput']):
    df = pd.read_csv(path)
    ts = df[column[0]]
    val = df[column[1]]
    return clean_coequipment(ts, val)

def read_vav_csv(path, column = ['PropertyTimestamp', 'AirFlowNormalized']):
    df = pd.read_csv(path)
    ts = df[column[0]]
    val = df[column[1]]
    return clean_coequipment(ts, val)

def read_facility_ahu(facility_id, ahu_list, max_length):
    ahu_data, label = [], []
    #print(max_length)
    #column = ['PropertyTimestamp', ahu_s]
    path = "/localtmp/split/ahu_property_file_" + str(facility_id) + "/"
    pops = []
    #print(column)
    for i, name in enumerate( ahu_list[facility_id]):
        if os.path.exists(path + name + '.csv') == False:
            continue
        ahu_d, flag = read_ahu_csv('/localtmp/split/ahu_property_file_'+str(facility_id)+'/' + name + '.csv')
        if flag and len(ahu_d) >= max_length:
            ahu_data.append(ahu_d[0:max_length])
            label.append((facility_id, name))
            print(facility_id, name)
        else:
            pops.append(name)

    for name in pops:
        ahu_list[facility_id].pop(ahu_list[facility_id].index(name))
    return ahu_data, label

def read_facility_vav(facility_id, mapping, max_length, ahu_list):
    vav_data, label = [], []
    #column = ['PropertyTimestamp', vav_s]
    path = "/localtmp/split/vav_box_property_file_" + str(facility_id) + "/"
    #print(column)
    for name in mapping[facility_id].keys():
        if os.path.exists(path + name + '.csv') == False:
            continue
        if mapping[facility_id][name] not in ahu_list[facility_id]:
            continue
            
        vav_d, flag = read_vav_csv('/localtmp/split/vav_box_property_file_'+str(facility_id)+'/' + name + '.csv')
        if flag and len(vav_d) >= max_length:
            vav_data.append(vav_d[0:max_length])
            label.append((facility_id, name))
            print(facility_id, name)
    return vav_data, label

def read_coequipment_ground_truth(path = './mapping_data.xlsx'):
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
        if 'AHU-13  Area 112  MP581-4-4-2' == child_name:
            print("removed ")
            continue
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
    #return value
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
                if config.norm_startpoint:
                    pass # not implemented by now
                else:
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
        #print("ohno")
        #print(len(val))
        return ts[0:maxl], val[0:maxl]
    else:
        #print("ohno")
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
    if config.sub_sample == True:
        return sub_sample(timestamps, vals, config)
    else:
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

def cross_validation_sample(total_cnt, test_cnt):
    if total_cnt % test_cnt != 0:
        total_cnt = test_cnt * int(total_cnt / test_cnt)
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

def STFT(x, config):
    fft_x = []
    for i in range(len(x)):
        fft_x.append(fft(x[i], config))
    return fft_x

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

def split_coequipment_train(vav_x, vav_y, test_index, train, test):
    train_vav, test_vav = [], []
    train_y, test_y = [], []
    shuffled_idx = np.arange(len(vav_x))
    np.random.shuffle(shuffled_idx)
    for i in shuffled_idx:
        if i in test_index:
            if vav_y[i][0] != test:
                continue
            test_vav.append(vav_x[i])
            test_y.append(vav_y[i])
        else:
            if vav_y[i][0] != train:
                continue
            train_vav.append(vav_x[i])
            train_y.append(vav_y[i])
    return train_vav, train_y, test_vav, test_y

# i anchor, k positive, j negative

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

def gen_coequipment_triplet(ahu_x, ahu_y, vav_x, vav_y, mapping):
    # 1: a, p, n = (vav, ahu, ahu)
    # 2: a, p, n = (vav, ahu, vav) 
    # 3: a, p, n = (vav, vav, ahu)
    # 4: a, p, n = (vav, vav, vav) 
    # 5: a, p, n = (ahu, vav, vav)
    triplet = []
    # 1 (vav, ahu, ahu)
    #print(mapping[vav_y[0][0]])
    for i in range(len(vav_y)): # anchor
        #print(mapping[vav_y[i][0]][vav_y[i][1]])
        k = ahu_y.index((vav_y[i][0], mapping[vav_y[i][0]][vav_y[i][1]])) # positive
        for j in range(len(ahu_y)): # negative
            if vav_y[i][0] != ahu_y[k][0] or vav_y[i][0] != ahu_y[j][0]:
                continue
            if j == k:
                continue
            #print(vav_y[i], ahu_y[k], ahu_y[j])
            sample = []
            sample.append(vav_x[i])
            sample.append(ahu_x[k])
            sample.append(ahu_x[j])
            triplet.append(sample)

    # 2 (vav, ahu, vav)
    '''
    for i in range(len(vav_y)): # anchor
        k = ahu_y.index((vav_y[i][0], mapping[vav_y[i][0]][vav_y[i][1]])) # positive
        for j in range(len(vav_y)): # negative
            if vav_y[i][0] != ahu_y[k][0] or vav_y[i][0] != vav_y[j][0]:
                continue
            if mapping[vav_y[i][0]][vav_y[i][1]] == mapping[vav_y[j][0]][vav_y[j][1]]:
                continue
            #print(vav_y[i], ahu_y[k], vav_y[j])
            sample = []
            sample.append(vav_x[i])
            sample.append(ahu_x[k])
            sample.append(vav_x[j])
            triplet.append(sample)
    

    # 3 (vav, vav, ahu)
    
    for i in range(len(vav_y)): # anchor
        for k in range(len(vav_y)): # positive
            for j in range(len(ahu_y)): # negative
                if vav_y[i][0] != vav_y[k][0] or vav_y[i][0] != ahu_y[j][0]:
                    continue
                if i == k or mapping[vav_y[i][0]][vav_y[i][1]] != mapping[vav_y[k][0]][vav_y[k][1]]:
                    continue
                if mapping[vav_y[i][0]][vav_y[i][1]] == ahu_y[j][1]:
                    continue
                #print(vav_y[i], vav_y[k], ahu_y[j])
                sample = []
                sample.append(vav_x[i])
                sample.append(vav_x[k])
                sample.append(ahu_x[j])
                triplet.append(sample)
    
    # 4 (vav, vav, vav)

    # 5 (ahu, vav, vav)
    
    for i in range(len(ahu_y)): # anchor
        for k in range(len(vav_y)): # positive
            for j in range(len(vav_y)): # negative
                if ahu_y[i][0] != vav_y[k][0] or ahu_y[i][0] != vav_y[j][0]:
                    continue
                if mapping[vav_y[k][0]][vav_y[k][1]] != ahu_y[i][1]:
                    continue
                if mapping[vav_y[j][0]][vav_y[j][1]] == ahu_y[i][1]:
                    continue
                #print(ahu_y[i], vav_y[k], vav_y[j])
                sample = []
                sample.append(ahu_x[i])
                sample.append(vav_x[k])
                sample.append(vav_x[j])
                triplet.append(sample)
    ''' 
    return triplet

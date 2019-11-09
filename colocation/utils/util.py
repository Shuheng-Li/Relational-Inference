import os
import csv
import yaml



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_config(path):
    return AttrDict(yaml.load(open(path, 'r'), Loader = yaml.FullLoader))

def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s)
    return write_log


def logging_result(file):
    def write_log(s):
        with open(file, 'a') as f:
            f.write(s)
    return write_log


def cal_sensor_acc(best_solution, test_y):
    total, cnt = 0, 0
    for i in range(len(best_solution)):
        for j in range(len(best_solution[i]) - 1):
            for k in range(j + 1, len(best_solution[i])):
                print(best_solution[i][j], best_solution[i][k])
                if best_solution[i][j] in test_y or best_solution[i][k] in test_y:
                    total += 1
                    print(best_solution[i][j], best_solution[i][k])
                else:
                    continue
                if int(best_solution[i][j] / 4) ==  int(best_solution[i][k] / 4):
                    cnt += 1
    print(total, cnt)
    acc = cnt / total
    return acc

def cal_room_acc(best_solution):
    pp, pn, np, nn = 0, 0, 0, 0 #(ground_truth, prediction)
    for i in range(len(best_solution)):
        for j in range(len(best_solution[i]) - 1):
            for k in range(j + 1, len(best_solution[i])):
                if(int(best_solution[i][j] / 4) == int(best_solution[i][k] / 4)):
                    pp += 1
                else:
                    pn += 1
                    np += 1
    nn = (len(best_solution) * len(best_solution[0])) * (len(best_solution) * len(best_solution[0]) - 1) /2 - pp - pn - np
    recall = pp / (pp + pn)
    acc_room = 0
    for i in range(len(best_solution)):
        r_id = int(best_solution[i][0] / 4)
        for j in range(1, 5):
            if j == 4:
                acc_room += 1
                break
            if int(best_solution[i][j] / 4) != r_id:
                break
    room_wise_acc = acc_room / len(best_solution)
    confusion = [[pp, np], [pn, nn]]
    return recall, room_wise_acc


import argparse
import torch
import torch.nn.functional as F
import sys
import os
import random
import utils.util as utils
import numpy as np
from Data import *
from models import *
from losses import *
import scipy.io as scio
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

#euclidean_norm = lambda x, y: np.abs(x - y)
import time
torch.cuda.set_device(1)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='train_and_test.py')
    parser.add_argument('-config', default = 'coequipment', type =str)
    parser.add_argument('-task', default = 'coequipment', type = str,
                        choices=['colocation', 'coequipment'])
    parser.add_argument('-model', default='han', type=str,
                        choices=['han', 'basic'])
    parser.add_argument('-loss', default='triplet', type=str,
                        choices=['triplet', 'angular', 'softmax'])
    parser.add_argument('-seed', default=2, type=int,
                        help="Random seed")
    parser.add_argument('-log', default='coe', type=str,
                        help="Log directory")
    parser.add_argument('-facility', default=10320, type=int,
                        help="Log directory")
    parser.add_argument('-split',default='room', type=str,
                        help="split 1/5 sensors or rooms for test",
                        choices = ['room', 'sensor'])
    args = parser.parse_args()
    config = utils.read_config(args.config + '.yaml')
    return args, config

args, config = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


def set_up_logging():
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + 'no_name' + '/'
    else:
        log_path = config.log + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logging = utils.logging(log_path + 'log.txt')
    logging_result = utils.logging_result(log_path + 'record.txt')
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return logging, logging_result, log_path

logging, logging_result, log_path = set_up_logging()

def test_coequipment(model, ahu_x, ahu_y, vav_x, vav_y, mapping, test):
    model.eval()
    wrongs = dict()
    facilities = []
    for ahu in ahu_y:
        if ahu[0] not in facilities:
            facilities.append(ahu[0])
    acc = []

    for f_id in facilities:
        if f_id != test:
            continue
        mapping_fid = mapping[f_id]
        test_vav = []
        test_ahu = []
        test_ahu_y = []
        test_vav_y = []
        for i in range(len(ahu_y)):
            if ahu_y[i][0] == f_id:
                test_ahu.append(ahu_x[i])
                test_ahu_y.append(ahu_y[i][1])

        for i in range(len(vav_y)):
            if vav_y[i][0] == f_id:
                test_vav.append(vav_x[i])
                test_vav_y.append(vav_y[i][1])

        with torch.no_grad():
            vav_out = model(torch.from_numpy(np.array(test_vav)).cuda())
            ahu_out = model(torch.from_numpy(np.array(test_ahu)).cuda())

        vav_out = vav_out.cpu().tolist()
        ahu_out = ahu_out.cpu().tolist()
        total_pairs = [0 for i in range(len(ahu_out))]
        repeate = [0 for i in range(len(vav_out))]
        for i in range(len(vav_out)):
            total_pairs[test_ahu_y.index(mapping_fid[test_vav_y[i]])] += 1
        #print(total_pairs)
        total = len(vav_out)
        cnt = 0
        euclidean_norm = lambda x, y: np.abs(x - y)
        for i, vav_emb in enumerate(vav_out):
            min_dist = 0xffff
            min_idx = 0
            for j, ahu_emb in enumerate(ahu_out):
                dist = np.linalg.norm(np.array(ahu_emb) - np.array(vav_emb))
                if dist < min_dist: #and total_pairs[j] > 0:
                    min_dist = dist
                    min_idx = j
            total_pairs[min_idx] -= 1
            min_dist = 0xfff
            if mapping_fid[test_vav_y[i]] == test_ahu_y[min_idx]:
                cnt += 1
            else:
                if mapping_fid[test_vav_y[i]] in wrongs:
                    wrongs[mapping_fid[test_vav_y[i]]] += 1
                else:
                    wrongs[mapping_fid[test_vav_y[i]]] = 1
        logging("Fid: %d Acc: %f\n" % (f_id, cnt / total))
        acc.append(cnt/total)
    #print(wrongs)
    model.train()
    return acc[0], wrongs


def main(test, train):
    # read data & window FFT
    logging(str(time.localtime()))

    mapping, ahu_list = read_coequipment_ground_truth('./mapping_data.xlsx')
    if config.all_facilities:
        facilities = [test, train]
        ahu_x, ahu_y, vav_x, vav_y = [], [], [], []
        for f_id in facilities:
            f_ahu_x, f_ahu_y = read_facility_ahu(f_id, ahu_list,  config.max_length)
            f_vav_x, f_vav_y = read_facility_vav(f_id, mapping, config.max_length, ahu_list)
            ahu_x += f_ahu_x
            ahu_y += f_ahu_y
            vav_x += f_vav_x
            vav_y += f_vav_y

        ahu_x = STFT(ahu_x, config)
        vav_x = STFT(vav_x, config)
    else:
        ahu_x, ahu_y = read_facility_ahu(args.facility, ahu_list,config.max_length)
        vav_x, vav_y = read_facility_vav(args.facility, mapping, config.max_length, ahu_list)
        ahu_x = STFT(ahu_x, config)
        vav_x = STFT(vav_x, config)

    logging("AHU %d total sensors, %d frequency coefficients, %d windows\n" % (len(ahu_x), ahu_x[0].shape[0], ahu_x[0].shape[1]))
    logging("VAV %d total sensors, %d frequency coefficients, %d windows\n" % (len(vav_x), vav_x[0].shape[0], vav_x[0].shape[1]))
    # split training & testing
    test_indices = cross_validation_sample(len(vav_y), int(len(vav_y) / 5))

    print("test indices:\n", test_indices)
    epochs_acc = []
    total_wrongs = dict()
    for fold, test_index in enumerate(test_indices):
        epochs_acc.append([])
        logging("Now training fold: %d" %(fold))
        train_vav_x, train_vav_y, test_vav_x, test_vav_y = split_coequipment_train(vav_x, vav_y, test_index, train, test)
        train_x = gen_coequipment_triplet(ahu_x, ahu_y, train_vav_x, train_vav_y, mapping)
        test_y = test_vav_y
        total_triplets = len(train_x)
        logging("Total training triplets: %d\n" % (total_triplets))
        testahu = dict()
        for v in test_y:
            if mapping[v[0]][v[1]] not in testahu:
                testahu[mapping[v[0]][v[1]]] = 1
            else:
                testahu[mapping[v[0]][v[1]]] += 1

        if args.loss == 'triplet':
            criterion = tripletLoss(margin = 1).cuda()
        elif args.loss == 'angular':
            criterion = angularLoss(margin = 1).cuda()
        elif args.loss == 'softmax':
            criterion = softmaxtripletLoss().cuda()

        model = STN(config.dropout).cuda()

        if config.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = config.learning_rate, momentum = 0.9, weight_decay = config.weight_decay)
        elif config.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

        if config.grad_norm > 0:
            nn.utils.clip_grad_value_(model.parameters(), config.grad_norm)
            for p in model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -config.grad_norm, config.grad_norm))

        print("Model : ", model)
        print("Criterion : ", criterion)

        #model = torch.load(log_path + 'model.pkl')
        print("testahus :\n", testahu)
        for epoch in range(config.epoch):
            train_loader = torch.utils.data.DataLoader(train_x, batch_size = config.batch_size, shuffle = True)
            logging("Now training %d epoch ......\n" % (epoch + 1))
            total_triplet_correct = 0
            for step, batch_x in enumerate(train_loader):

                anchor = batch_x[0].cuda()
                pos = batch_x[1].cuda()
                neg = batch_x[2].cuda()

                output_anchor = model(anchor) 
                output_pos = model(pos) 
                output_neg = model(neg)

                loss, triplet_correct = criterion(output_anchor, output_pos, output_neg)
                total_triplet_correct += triplet_correct.item()

                optimizer.zero_grad()           
                loss.backward()                 
                optimizer.step()
                if step % 200 == 0 and step != 0:
                    logging("loss "+str(loss)+"\n")
                    logging("triplet_acc " + str(triplet_correct.item()/config.batch_size) + "\n")

            logging("Triplet accuracy: %f" % (total_triplet_correct/total_triplets))

            #torch.save(model, log_path + args.task + '_' + args.model + '_model.pkl')
            acc, wrongs = test_coequipment(model, ahu_x, ahu_y, test_vav_x, test_vav_y, mapping, test)
            epochs_acc[fold].append(acc)
            for keys in wrongs:
                if keys in total_wrongs:
                    total_wrongs[keys] += wrongs[keys]
                else:
                    total_wrongs[keys] = wrongs[keys]
            print("Wrong equipments : ")
            print(total_wrongs)

    overall_epoch_acc = [0 for i in range(config.epoch)]

    for i in range(len(epochs_acc)):
        for j in range(len(epochs_acc[0])):
            overall_epoch_acc[j] += epochs_acc[i][j] / len(epochs_acc)

    #print(overall_epoch_acc)
    logging("Best accuracy : %f, best epoch: %d\n" % (max(overall_epoch_acc), overall_epoch_acc.index(max(overall_epoch_acc))))
    return max(overall_epoch_acc)

if __name__ == '__main__':
    #main()
    
    tests = [10312, 10320, 10381, 10596, 10606, 10642]
    trains = [10312, 10320, 10381, 10596, 10606, 10642]
    for test in tests:
        for train in trains:
            if test == train:
                continue
            random.seed(args.seed)
            np.random.seed(args.seed)
            acc = main(test, train)
            logging("Test " + str(test) + " Train " + str(train) + " acc " + str(acc))

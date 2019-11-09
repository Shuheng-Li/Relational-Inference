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
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time
from GeneticAlgorithm.colocation import run
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

torch.cuda.set_device(1)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-config', default = 'stn', type =str)
    parser.add_argument('-task', default = 'colocation', type = str,
                        choices=['colocation', 'coequipment'])
    parser.add_argument('-model', default='stn', type=str,
                        choices=['stn'])
    parser.add_argument('-loss', default='angular', type=str,
                        choices=['triplet', 'angular'])
    parser.add_argument('-seed', default=2, type=int,
                        help="Random seed")
    parser.add_argument('-log', default='han', type=str,
                        help="Log directory")
    parser.add_argument('-facility', default=10606, type=int,
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

def test_coequipment(model, ahu_x, test_vav, ahu_y, test_y, mapping):
    model.eval()
    with torch.no_grad():
        vav_out = model(torch.from_numpy(np.array(test_vav)).cuda())
        ahu_out = model(torch.from_numpy(np.array(ahu_x)).cuda())

    vav_out = vav_out.cpu().tolist()
    ahu_out = ahu_out.cpu().tolist()

    total = len(vav_out)
    cnt = 0
    for i, vav_emb in enumerate(vav_out):
        min_dist = 0xffffff
        min_idx = 0
        for j, ahu_emb in enumerate(ahu_out):
            dist = np.linalg.norm(np.array(ahu_emb) - np.array(vav_emb))
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        if mapping[test_y[i]] == ahu_y[min_idx]:
            print(i, min_idx)
            cnt += 1
        print(test_y[i], ahu_y[min_idx])
    acc = cnt / total
    return acc


def test_colocation(test_x, test_y, model, fold, split):
    model.eval()
    
    with torch.no_grad():
        if args.model == 'stn':
            out = model(torch.from_numpy(np.array(test_x)).cuda())

        test_triplet = gen_colocation_triplet(test_x, test_y, prevent_same_type = True)
        test_loader = torch.utils.data.DataLoader(test_triplet, batch_size = 1, shuffle = False)
        cnt = 0
        for step, batch_x in enumerate(test_loader):
            if args.model == 'stn':
                anchor = batch_x[0].cuda()
                pos = batch_x[1].cuda()
                neg = batch_x[2].cuda()

            output_anchor = model(anchor) 
            output_pos = model(pos) 
            output_neg = model(neg)
            distance_pos = (output_anchor - output_pos).pow(2).sum(1).pow(1/2)
            distance_neg = (output_anchor - output_neg).pow(2).sum(1).pow(1/2)
            if distance_neg > distance_pos:
                cnt += 1

        logging("Testing triplet acc: %f" %(cnt / len(test_triplet)))
   
    test_out = out.cpu().tolist()
    test_corr = np.corrcoef(np.array(test_out))


    scio.savemat('./output/corr_' + str(fold) + '.mat', {'corr':test_corr})
    best_solution, acc, ground_truth_fitness, best_fitness = run.ga(path_m = './output/corr_' + str(fold) + '.mat', path_c = '10_rooms.json')
    recall, room_wise_acc = utils.cal_room_acc(best_solution)
    logging("recall = %f, room_wise_acc = %f:\n" %(recall, room_wise_acc))

    logging("Ground Truth Fitness %f Best Fitness: %f \n" % (ground_truth_fitness, best_fitness))
    logging("Edge-wise accuracy: %f \n" % (acc))

    model.train()
    return best_solution, recall, room_wise_acc


def main():

    # read data & STFT

    logging(str(time.localtime()))
    if args.task == 'colocation':
        x, y, true_pos = read_colocation_data(config)
        x = STFT(x, config)
        logging("%d total sensors, %d frequency coefficients, %d windows\n" % (len(x), x[0].shape[0], x[0].shape[1]))

    elif args.task == 'coequipment':
        pass

    if args.task == 'colocation':
        test_indexes = cross_validation_sample(50, 10)
    elif args.task == 'coequipment':
        pass

    print("test indexes:\n", test_indexes)

    fold_recall = []
    fold_room_acc = []

    for fold, test_index in enumerate(test_indexes):

        logging("Now training fold: %d" %(fold))

        # split training & testing

        if args.task == 'colocation':
            print("Test indexes: ", test_index)
            train_x, train_y, test_x, test_y = split_colocation_train(x, y, test_index, args.split)
            train_x = gen_colocation_triplet(train_x, train_y)
        elif args.task == 'coequipment':
            pass

        total_triplets = len(train_x)
        logging("Total training triplets: %d\n" % (total_triplets))
        print(test_y)
        
        if args.loss == 'triplet':
            criterion = tripletLoss(margin = 1).cuda()
        elif args.loss == 'angular':
            criterion = angularLoss(margin = 1).cuda()

        if args.model == 'stn':
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

        for epoch in range(config.epoch):
            train_loader = torch.utils.data.DataLoader(train_x, batch_size = config.batch_size, shuffle = True)
            logging("Now training %d epoch ......\n" % (epoch + 1))
            total_triplet_correct = 0
            for step, batch_x in enumerate(train_loader):

                if args.model == 'stn':
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

            torch.save(model, log_path + args.task + '_' + args.model + '_model.pkl')

            if args.task == 'colocation':
                solution, recall, room_wise_acc = test_colocation(test_x, test_y, model, fold, args.split)
                solution = solution.tolist()

                logging_result("fold: %d, epoch: %d\n" % (fold, epoch))
                logging_result("Acc: %f\n" %(recall))
                logging("fold: %d, epoch: %d\n" % (fold, epoch))
                logging("Acc: %f\n" %(recall))

                for k in range(len(solution)):
                    for j in range(len(solution[k])):
                        logging_result(str(solution[k][j]) + ' ')
                    logging_result('\n')
                logging_result('\n')

            elif args.task == 'coequipment':
                pass

        fold_recall.append(recall)
        fold_room_acc.append(room_wise_acc)

    
    logging("Final recall : %f \n" % (np.array(fold_recall).mean()))
    logging("Final room accuracy : %f \n" % (np.array(fold_room_acc).mean()))

if __name__ == '__main__':
    main()
    
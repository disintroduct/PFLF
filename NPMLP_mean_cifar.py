#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
import argparse
import matplotlib
import sys
import pylab
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
import math
import numpy as np
import random
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import autograd
from tensorboardX import SummaryWriter
from sympy import solve
from sympy.abc import y
import sympy as sy
from scipy import optimize

from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from options import args_parser
from Update import LocalUpdate
from FedNets import MLP1, CNNMnist, CNN_test
from averaging import average_weights
from Privacy import Privacy_account, Adjust_T, Noise_TB_decay
from Noise_add import noise_add, users_sampling, clipping

from torchvision.datasets import cifar
import torchvision.transforms as transf


def main(args):
    # ####-Choose Variable-#####
    set_variable = args.set_num_Chosenusers
    set_variable0 = copy.deepcopy(args.set_epochs)
    set_variable1 = copy.deepcopy(args.set_privacy_budget)

    if not os.path.exists('./experiresult/cifar'):
        os.mkdir('./experiresult/cifar')
    print('1')
    # load dataset and split users
    # 0.1307和0.3081是mnist数据集的均值和标准差，因为mnist数据值都是灰度图，
    # 所以图像的通道数只有一个，因此均值和标准差各一个
    # trans = transf.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    transforms = transf.Compose([transf.ToTensor(),transf.Normalize([0.5],[0.5])])

    dataset_train, dataset_test = [], []
    dataset_train = cifar.CIFAR10('./dataset/cifar/', train=True, download=True,
                                   transform=transforms)
    dataset_test = datasets.CIFAR10('./dataset/cifar/', train=False, download=True,
                                  transform=transforms)
    print('2')
    # sample users
    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
        # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test) 
        dict_sever = cifar_iid(dataset_test, args.num_users)
    else:
        dict_users = cifar_noniid(dataset_train, args.num_users)
        dict_sever = cifar_noniid(dataset_test, args.num_users)

    img_size = dataset_train[0][0].shape
    print('3')
    for v in range(len(set_variable)):
        # 加入离线率系统统计
        args.offline_rate = set_variable[v] / args.num_users

        final_train_loss = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_train_accuracy = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_test_loss = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_test_accuracy = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_com_cons = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        args.num_Chosenusers = copy.deepcopy(set_variable[v])
        for s in range(len(set_variable0)):
            for j in range(len(set_variable1)):
                args.epochs = copy.deepcopy(set_variable0[s])
                # args.privacy_budget = copy.deepcopy(set_variable1[j])
                args.privacy_budget = copy.deepcopy(set_variable1[j])
                print("dataset:", args.dataset, " num_users:", args.num_users, " num_chosen_users:", args.num_Chosenusers, " Privacy budget:", args.privacy_budget,\
                      " epochs:", args.epochs, "local_ep:", args.local_ep, "local train size", args.num_items_train, "batch size:", args.local_bs)        
                loss_test, loss_train = [], []
                acc_test, acc_train = [], []          
                for m in range(args.num_experiments):
                    # build model
                    net_glob = None
                    if args.model == 'cnn' and args.dataset == 'mnist':
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            # net_glob = CNNMnist(args=args).cuda()
                            net_glob = CNN_test(args=args).cuda()
                        else:
                            net_glob = CNNMnist(args=args)
                    elif args.model == 'mlp':
                        len_in = 1
                        for x in img_size:
                            len_in *= x
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            net_glob = MLP1(dim_in=len_in, dim_hidden=256,\
                                            dim_out=args.num_classes).cuda()
                        else:
                            net_glob = MLP1(dim_in=len_in, dim_hidden=256,\
                                            dim_out=args.num_classes)
                    else:
                        exit('Error: unrecognized model')
                    print("Nerual Net:", net_glob)

                    net_glob.train()  #Train() does not change the weight values
                    # copy weights
                    w_glob = net_glob.state_dict()
                    w_size = 0
                    w_size_all = 0
                    for k in w_glob.keys():
                        size = w_glob[k].size()
                        if(len(size) == 1):
                            nelements = size[0]
                        else:
                            nelements = size[0] * size[1]
                        w_size += nelements*4
                        w_size_all += nelements
                        # print("Size ", k, ": ",nelements*4)
                    print("Weight Size:", w_size, " bytes")
                    print("Weight & Grad Size:", w_size*2, " bytes")
                    print("Each user Training size:", 784* 8/8* args.local_bs, " bytes")
                    print("Total Training size:", 784 * 8 / 8 * 60000, " bytes")
                    # training
                    threshold_epochs = copy.deepcopy(args.epochs)          
                    threshold_epochs_list, noise_list = [], []
                    loss_avg_list, acc_avg_list, list_loss, loss_avg = [], [], [], []  
                    eps_tot_list, eps_tot = [], 0
                    com_cons = []
                    # ##  FedAvg Aglorithm  ###
                    # ## Compute noise scale ###
                    noise_scale = copy.deepcopy(Privacy_account(args,\
                                            threshold_epochs, noise_list, 0))
                    for iter in range(args.epochs):
                        print('\n', '*' * 20, f'Epoch: {iter}', '*' * 20)
                        start_time = time.time()
                        if args.num_Chosenusers < args.num_users:
                            chosenUsers = random.sample(range(1, args.num_users)\
                                                        , args.num_Chosenusers)
                                                        # ,np.random.poisson(lam=args.num_Chosenusers))
                            chosenUsers.sort()
                        else:
                            chosenUsers = range(args.num_users)
                        print("\nChosen users:", chosenUsers)
                        w_locals, w_locals_1ep, loss_locals, acc_locals = [], [], [], []
                        for idx in range(len(chosenUsers)):
                            local = LocalUpdate(args=args, dataset=dataset_train,\
                                    idxs=dict_users[chosenUsers[idx]], tb=summary)
                            w_1st_ep, w, loss, acc = local.update_weights(\
                                                    net=copy.deepcopy(net_glob))
                            w_locals.append(copy.deepcopy(w))
                            ### get 1st ep local weights ###
                            w_locals_1ep.append(copy.deepcopy(w_1st_ep))            
                            loss_locals.append(copy.deepcopy(loss))
                            # print("User ", chosenUsers[idx], " Acc:", acc, " Loss:", loss)
                            acc_locals.append(copy.deepcopy(acc))
  
                        ### Clipping ###
                        for idx in range(len(chosenUsers)):
                            w_locals[idx] = copy.deepcopy(clipping(args, w_locals[idx]))
                            # print(get_2_norm(w_locals[idx], w_glob))
  
                        ### perturb 'w_local' ###
                        w_locals = noise_add(args, noise_scale, w_locals)                        

                        ### update global weights ###                
                        # w_locals = users_sampling(args, w_locals, chosenUsers)
                        w_glob = average_weights(w_locals) 
                         
                        # copy weight to net_glob
                        net_glob.load_state_dict(w_glob)
                        # global test
                        list_acc, list_loss = [], []
                        net_glob.eval()
                        for c in range(args.num_users):
                            net_local = LocalUpdate(args=args,dataset=dataset_test,\
                                                    idxs=dict_sever[idx], tb=summary)
                            acc, loss = net_local.test(net=net_glob)                    
                            # acc, loss = net_local.test_gen(net=net_glob,\
                            # idxs=dict_users[c], dataset=dataset_test)
                            list_acc.append(acc)
                            list_loss.append(loss)
                        # print("\nEpoch:{},Global test loss:{}, Global test acc:{:.2f}%".\
                        #      format(iter, sum(list_loss) / len(list_loss),\
                        #      100. * sum(list_acc) / len(list_acc)))
                        
                        # print loss
                        loss_avg = sum(loss_locals) / len(loss_locals)
                        acc_avg = sum(acc_locals) / len(acc_locals)
                        loss_avg_list.append(loss_avg)
                        acc_avg_list.append(acc_avg) 
                        print("\nTrain loss: {}, Train acc: {}".\
                              format(loss_avg_list[-1], acc_avg_list[-1]))
                        print("\nTest loss: {}, Test acc: {}".\
                              format(sum(list_loss) / len(list_loss),\
                                     sum(list_acc) / len(list_acc)))
                        
                        noise_list.append(noise_scale)
                        threshold_epochs_list.append(threshold_epochs)
                        print('\nNoise Scale:', noise_list)
                        print('\nThreshold epochs:', threshold_epochs_list)
                        """
                        if args.dp_mechanism == 'CRD' and iter >= 1:
                            threshold_epochs = Adjust_T(args, loss_avg_list,\
                                                threshold_epochs_list, iter)
                            noise_scale = copy.deepcopy(Privacy_account(args,\
                                        threshold_epochs, noise_list, iter))
                        """
                        if args.dp_mechanism == 'NSD' and iter >= 1:
                            noise_scale,eps_tot= Noise_TB_decay(args, noise_list,\
                                    loss_avg_list, args.dec_cons, iter, 'UD')
                            noise_list_next = copy.deepcopy(noise_list)
                            noise_list_next.append(noise_scale)
                            _,eps_tot_next= Noise_TB_decay(args, noise_list_next,\
                                  loss_avg_list, args.dec_cons, iter+1, 'UD')
                            
                            eps_tot_list.append(eps_tot)
                        
                            if eps_tot_next >= args.privacy_budget:
                                threshold_epochs = 0
                            print('\nTotal eps:',eps_tot_list,eps_tot_next)
                            
                        # print run time of each experiment
                        end_time = time.time()
                        print('Run time: %f second' % (end_time - start_time))
                    
                        # if iter >= threshold_epochs: break
                    loss_train.append(loss_avg)
                    acc_train.append(acc_avg)
                    loss_test.append(sum(list_loss) / len(list_loss))
                    acc_test.append(sum(list_acc) / len(list_acc))
                    com_cons.append(iter+1)
                    
                # record results
                final_train_loss[s][j]=copy.deepcopy(sum(loss_train)/len(loss_train))
                final_train_accuracy[s][j]=copy.deepcopy(sum(acc_train)/len(acc_train))
                final_test_loss[s][j]=copy.deepcopy(sum(loss_test)/len(loss_test))
                final_test_accuracy[s][j]=copy.deepcopy(sum(acc_test)/len(acc_test))
                final_com_cons[s][j]=copy.deepcopy(sum(com_cons)/len(com_cons))
    
            print('\nFinal train loss:', final_train_loss)
            print('\nFinal train acc:', final_train_accuracy)
            print('\nFinal test loss:', final_test_loss)
            print('\nFinal test acc:', final_test_accuracy)
            
        timeslot = int(time.time())
        data_test_loss = pd.DataFrame(index = set_variable0, columns =\
                                      set_variable1, data = final_train_loss)
        data_test_loss.to_csv('./experiresult/cifar/'+'train_loss_{}_{}_{}.csv'.\
                              format(set_variable[v],args.dp_mechanism,timeslot))
        data_test_loss = pd.DataFrame(index = set_variable0, columns =\
                                      set_variable1, data = final_test_loss)
        data_test_loss.to_csv('./experiresult/cifar/'+'test_loss_{}_{}_{}.csv'.\
                              format(set_variable[v],args.dp_mechanism,timeslot))
        data_test_acc = pd.DataFrame(index = set_variable0, columns =\
                                     set_variable1, data = final_train_accuracy)
        data_test_acc.to_csv('./experiresult/cifar/'+'train_acc_{}_{}_{}.csv'.\
                             format(set_variable[v],args.dp_mechanism,timeslot))
        data_test_acc = pd.DataFrame(index = set_variable0, columns =\
                                     set_variable1, data = final_test_accuracy)
        data_test_acc.to_csv('./experiresult/cifar/'+'test_acc_{}_{}_{}.csv'.\
                             format(set_variable[v],args.dp_mechanism,timeslot))
        data_test_acc = pd.DataFrame(index = set_variable0, columns =\
                                     set_variable1, data = final_com_cons)
        data_test_acc.to_csv('./experiresult/cifar/'+'aggregation_consuming_{}_{}_{}_{}_{}.csv'.\
                             format(set_variable[v],args.dp_mechanism,timeslot,args.offline_rate,args.num_users))

# 均匀分布
def poission_chosenuser():
    chose_list = [30, 90, 150, 210, 270]
    s = []
    for i in chose_list:
        j = np.random.poisson(lam=i)
        s.append(j)
    return s


if __name__ == '__main__':    
    # return the available GPU
    av_GPU = torch.cuda.is_available()
    if  av_GPU == False:
        exit('No available GPU')
    # parse args    
    args = args_parser()
    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')
	### differential privacy ###
    args.dp_mechanism = 'NSD' ### NSD, CRD or non-CRD###
    args.dec_cons = 0.8 ## discount constant
    # args.privacy_budget = 100
    args.delta = 0.01
    # 离线率    
    # arg.offline_rate = 1

    args.gpu = -1               # -1 (CPU only) or GPU = 0
    args.lr = 0.01          # 0.001 for cifar dataset
    args.model = 'mlp'         # 'mlp' or 'cnn'
    args.dataset = 'cifar'     # 'mnist'

    args.num_Chosenusers = 50
    args.epochs = 100           # numb of global iters
    args.local_ep = 5        # numb of local iters
    args.num_items_train = 800 # numb of local data size # 
    args.num_items_test =  512
    args.local_bs = 128         ### Local Batch size (1200 = full dataset ###
                               ### size of a user for mnist, 2000 for cifar) ###
                               
    # args.set_privacy_budget = range(8,24,4)
    # args.set_privacy_budget = range(4, 14, 2)
    # args.set_privacy_budget = [10]
    
    # 全局迭代次数设置
    # args.set_epochs = range(100,305,50)
    # args.set_epochs = [25, 50, 75, 100, 125, 150]
    # args.set_epochs = range(10, 51, 10)
    args.set_epochs = [10]
    # args.set_num_Chosenusers = [50]
    ### numb of users ###
    # args.num_users = 50         
    # args.set_num_Chosenusers = [10, 20, 30, 40]
    args.set_dec_cons = [0.7,0.75,0.80,0.85,0.9,0.95]
    num_list = [1, 2, 3, 4, 5]

    # offline_rate = [0.3, 0.5, 0.7, 0.9, 1]
    # offline_rate = [0.2, 0.4, 0.6, 0.8, 1]

    # 不同隐私预算
    # args.set_privacy_budget = range(2, 15, 2)
    args.set_privacy_budget = [10000]


    args.num_experiments = 1
    args.clipthr = 20
    print("Begining")
    args.iid = True

    args.num_users = 300  
    # num_list_users = [100,150,200,250,300]
    num_list_users = [300]
    chose_list = [30, 90, 150, 210, 270]
    # 均匀分布   
    # args.set_num_Chosenusers = [30, 90, 150, 210, 270]
    # possion distribution
    # args.set_num_Chosenusers = poission_chosenuser()
    args.set_num_Chosenusers = [30]
    print(args.set_num_Chosenusers)
    # 泊松分布
    # args.set_num_Chosenusers = np.random.poisson(lam=30, size=2)
    # ni = 0
    #for i in range (300):
    #    if random.random() > 0.9:
    #        ni = ni + 1
    # args.set_num_Chosenusers = ni
    #args.set_num_Chosenusers = [ni]

    localtime_0 = time.asctime(time.localtime(time.time()))
    main(args)
    localtime = time.asctime(time.localtime(time.time()))

    # args.set_num_Chosenusers = [30,90,150,210,270]
    # args.set_num_Chosenusers = [10,20,30,40,50]

    # args.offline_rate = offline_rate[0]

    # for i in num_list_users:
    #     args.num_users = i
    #     localtime_0 = time.asctime(time.localtime(time.time()))
    #     main(args)
    #     localtime = time.asctime(time.localtime(time.time()))

    print("Beginning time is: ", localtime_0)
    print("Local time is:", localtime)



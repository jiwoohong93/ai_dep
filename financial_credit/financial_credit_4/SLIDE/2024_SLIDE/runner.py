import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from train import *
from slide import fair_penalty
from load_data import *
from utils import *
from updates import *

# torch CPU restriction
torch.set_num_threads(4)

# Model defined

class DNN(nn.Module) :

    def __init__(self, dimension) :

        super(DNN, self).__init__()
        self.dimension = dimension
        self.hidden = 100
        self.fc1 = nn.Linear(self.dimension, self.hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden, 2)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, inputs) :

        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        logits = self.fc2(outputs)
        probs = self.softmax(logits)

        return logits, probs

class Linear(nn.Module) :

    def __init__(self, dimension) :

        super(Linear, self).__init__()
        self.dimension = dimension
        self.fc1 = nn.Linear(self.dimension, 2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, inputs) :

        logits = self.fc1(inputs)
        probs = self.softmax(logits)

        return logits, probs



# Run SLIDE + DI

def runner(args) :

    print("=================================================================================")

    # fix seed for all
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # when is today?
    from datetime import datetime
    today = datetime.today()
    date = today.strftime("%Y%m%d")

    # device
    device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.cuda("cpu")

    
    # load data
    if args.dataset == "toy" :
        (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test) = load_toy_dataset()
    elif args.dataset == "law" :
        (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test) = load_law_dataset()
    else :
        raise NotImplementedError

    # to tensor    
    xs_train, x_train, y_train, s_train = torch.from_numpy(xs_train), torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(s_train)
    xs_test, x_test, y_test, s_test = torch.from_numpy(xs_test), torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(s_test)
    
    test_set = TensorDataset(x_test, y_test, s_test)
    test_batch_size = len(test_set)
    testloader_0_ = DataLoader(test_set, batch_size = test_batch_size, shuffle = False)
    sensitives = s_test
    
    # define the dimension
    d = x_train.shape[1]
    assert d == x_test.shape[1]

    # define model to use
    if args.model_type == "dnn" :
        model = DNN(dimension = d)
        model.to(device)
    elif args.model_type == "linear" :
        model = Linear(dimension = d)
        model.to(device)
    else :
        print("DNN or Linear model only provided")
        raise NotImplementedError

    # define optimizer to use
    if args.opt == "sgd" :
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = 0.0, weight_decay = 0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.9)
    elif args.opt == "adam" :
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma = 0.5)
    else :
        print("only SGD and Adam optimizer to use!")
        raise NotImplementedError

    # criterions
    util_criterion = nn.CrossEntropyLoss()
    fair_criterion = fair_penalty(mode = args.mode, gamma = 0.5, tau = args.tau)

    # learning
    train_losses = []
    for epoch in tqdm(range(args.epochs), desc = "DI learning") :

        learning_stats = {"acc" : [], "bacc" : [], "di" : []}

        # train for one epoch
        model.train()
        train_loss = train_full_batch_di(model, x_train, y_train, s_train, optimizer, scheduler, args.batch_size, args.lmda, args.tau, util_criterion, fair_criterion, device)

        train_losses.append(train_loss)

        # print at each 500 epoch
        if (epoch + 1) % 500 == 0 or epoch == args.epochs - 1:

            print("+"*20)
            print("Epoch: {}".format(epoch + 1))
        
            # test at the last
            model.eval()
            with torch.no_grad() :
                preds_, all_targets, all_sensitives = test_(model, testloader_0_, device)

                test_perfs = evaluate_di(preds_, all_targets, sensitives) # preds_ contains 6 tuples

            # update learning stats
            learning_stats = update_perfs_di(test_perfs, learning_stats)

            # print at cmd
            print("Dataset : {}, Mode : {}, Lambda : {}, Tau : {}".format(args.dataset, args.mode, 
                                                                          args.lmda, args.tau))
            print("(acc,  bacc,  di)")
            print(test_perfs)

    # save performances
    options = "{}/".format(date)
    file_name = options + "perfs_di_lr{}_epochs{}_opt{}.csv".format(args.lr, args.epochs, args.opt)
    if not os.path.exists(args.result_path + options) :
        os.makedirs(args.result_path + options)    
    write_perfs_di(args.result_path, file_name, args.mode, args.lmda, args.tau, learning_stats)


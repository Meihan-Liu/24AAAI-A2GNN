import time
import glob
import argparse
import itertools
import os
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.transforms import Constant

from model import Model
from utils import evaluate, CitationDataset, TwitchDataset
from layers import GradReverse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=200, 
                    help='random seed')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--dropout_ratio', type=float, default=0.1, 
                    help='dropout ratio')
parser.add_argument('--nhid', type=int, default=128, 
                    help='hidden size')
parser.add_argument('--patience', type=int, default=100, 
                    help='patience for early stopping')
parser.add_argument('--device', type=str, default='cuda:1', 
                    help='specify cuda devices')
parser.add_argument('--run_times', type=int, default=1, 
                    help='run times')
parser.add_argument('--epochs', type=int, default=200, 
                    help='maximum number of epochs')
parser.add_argument('--source', type=str, default='DBLPv7', 
                    help='source domain data')
parser.add_argument('--target', type=str, default='ACMv9', 
                    help='target domain data')
parser.add_argument('--weight', type=float, default=0.001, 
                    help='trade-off parameter')
parser.add_argument('--source_pnum', type=int, default=0, 
                    help='the number of propagation layers on the source graph')
parser.add_argument('--target_pnum', type=int, default=30, 
                    help='the number of propagation layers on the target graph')
args = parser.parse_args()
print(args)


if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', 
                    args.source)
    source_dataset = CitationDataset(path, args.source)
if args.source in {'EN', 'DE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', 
                    args.source)
    source_dataset = TwitchDataset(path, args.source)
if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', 
                    args.target)
    target_dataset = CitationDataset(path, args.target)
if args.target in {'EN', 'DE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', 
                    args.target)
    target_dataset = TwitchDataset(path, args.target)
source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)

args.num_classes = len(np.unique(source_dataset[0].y.numpy()))    
args.num_features = source_data.x.size(1)
args.save_path = './'   


def train(args, source_data, target_data):    
    min_loss = 1e10
    patience_cnt = 0
    loss_values = []
    best_epoch = 0
    
    model = Model(args).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, momentum=args.momentum, 
                                weight_decay=args.weight_decay)
    
    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        correct = 0
        
        # Source Domain Cross-Entropy Loss
        source_plogits = model(source_data.x, source_data.edge_index, 
                               args.source_pnum) 
        target_plogits = model(target_data.x, target_data.edge_index, 
                               args.target_pnum) 
        train_loss = F.nll_loss(F.log_softmax(source_plogits, dim=1),
                                source_data.y)
        loss = train_loss
        
        # ADV Loss
        p = float(epoch) / args.epochs
        GradReverse.rate = 2. / (1. + np.exp(-10. * p)) - 1
        
        source_feature = model.feat_bottleneck(source_data.x, source_data.edge_index,
                                               args.source_pnum)
        target_feature = model.feat_bottleneck(target_data.x, target_data.edge_index,
                                               args.target_pnum)
        
        source_dlogits = model.domain_classifier(source_feature, 
                                                 source_data.edge_index)
        target_dlogits = model.domain_classifier(target_feature, 
                                                 target_data.edge_index)
        
        domain_label = torch.tensor([0] * source_feature.shape[0] +\
                                    [1] * target_feature.shape[0]).to(args.device)
        
        domain_loss = F.cross_entropy(torch.cat([source_dlogits, target_dlogits],\
                                                0), domain_label)
        
        # Overall Loss
        loss = loss + args.weight * domain_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            acc, _, _, _= evaluate(source_data, model)
            _, macro_f1, micro_f1, test_loss = evaluate(target_data, model, 
                                                        args.target_pnum)
            
            print('Epoch: {:04d}'.format(epoch + 1), 
                  'train_loss: {:.6f}'.format(loss), 
                  'test_loss: {:.6f}'.format(test_loss), 
                  'train_acc: {:.6f}'.format(acc), 
                  'macro_f1: {:.6f}'.format(macro_f1), 
                  'micro_f1: {:.6f}'.format(micro_f1))
        
        loss_values.append(loss)
        torch.save(model.state_dict(), args.save_path+'{}.pth'.format(epoch))
        
        if loss_values[-1] < min_loss:
            min_loss = loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob(args.save_path+'*.pth')
        for f in files:
            epoch_nb = int(f[2:].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)
                pass

    files = glob.glob(args.save_path+'*.pth')
    for f in files:
        epoch_nb = int(f[2:].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    time_use = time.time() - t
    
    return model, best_epoch, time_use
    
    
# Run Experiment
macro_f1_dict = []
micro_f1_dict = []
all_dict = []
time_dict = []

for i in range(args.run_times):
    model, best_model, time_use = train(args, source_data, target_data)
    model.load_state_dict(torch.load(args.save_path+'{}.pth'.format(best_model)))
    acc, _, _, _ = evaluate(source_data, model)
    _, macro_f1, micro_f1, test_loss = evaluate(target_data, model, 
                                                args.target_pnum)

    print('{} -> {}  source acc = {:.6f}, macro_f1 = {:.6f}, \
    micro_f1 = {:.6f}'.format(args.source, args.target, acc, macro_f1, micro_f1))
    macro_f1_dict.append(macro_f1)
    micro_f1_dict.append(micro_f1)
    all_dict.append(macro_f1)
    all_dict.append(micro_f1)
    time_dict.append(time_use)

macro_f1_dict_print = [float('{:.6f}'.format(i)) for i in macro_f1_dict]
micro_f1_dict_print = [float('{:.6f}'.format(i)) for i in micro_f1_dict]
all_dict_print = [float('{:.6f}'.format(i)) for i in all_dict]

print('mAcro:', macro_f1_dict_print, 
      'mean {:.4f}'.format(np.mean(macro_f1_dict)), 
      ' std {:.4f}'.format(np.std(macro_f1_dict)))
print('mIcro:', micro_f1_dict_print,
      'mean {:.4f}'.format(np.mean(micro_f1_dict)),
      ' std {:.4f}'.format(np.std(micro_f1_dict)))

print('mAcro-mIcro:', all_dict_print)
print('time use mean {:.4f}'.format(np.mean(time_dict)), 
      ' std {:.4f}'.format(np.std(time_dict)))

    
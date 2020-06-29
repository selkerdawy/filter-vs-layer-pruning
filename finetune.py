from __future__ import print_function
import argparse
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pdb
import cifar as models


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=164, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr-decay-every', type=int, default=100,  help='learning rate decay by 10 every X epochs')
parser.add_argument('--lr-decay-scalar', type=float, default=0.1,
                    help='--')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--gpuid', default=0, type=int, help='')
parser.add_argument('--action', default='train', type=str, help='')

#Layer pruning
parser.add_argument('--load-model', default='', type=str, help='pretrained model')
parser.add_argument('--criterion', default='', type=str, help='Path to criterion')
parser.add_argument('--remove-layers', default=0, type=int, help='How many layers/blocks to remove')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

arch_method_blocksfn_mapping = {}

def get_pruned_resnet56(model, crit, groups):

    groupscum = []
    sofar=0
    for g in groups:
        groupscum += [sofar+g]
        sofar+=g

    mapping = {0: model.layer1, 1:model.layer2, 2:model.layer3}
    sortedidx = np.argsort(crit)
    j = 0
    i = 0
    while i < args.remove_layers:
        blockid = sortedidx[j] - 1
        for whichlayer, g in enumerate(groupscum):
            if g > blockid:
                break

        whichblock = blockid%groups[whichlayer]
        if (whichblock > 0 or whichlayer ==0 ) and blockid >= 0 :
            block = mapping[whichlayer][whichblock]
            print('       Removing block %d from group %d'%(whichblock,whichlayer+1))
            mapping[whichlayer][whichblock] = nn.Identity()
            i+=1
        j+=1


def prune_cifar_resnet56(model):
    import pickle
    crit = pickle.load(open(args.criterion, 'rb'))
    depth = args.depth
    groups = [(depth - 2) // 6]*3
    get_pruned_resnet56(model, crit, groups)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

print('==> Model ', args.arch)
from thop import profile
model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, add_gates = False)

if len(args.load_model)>0:
    checkpoint = torch.load(args.load_model)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    #checkpoint = {k.replace('features','feature').replace('module.',''):v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)

args.arch_depth = args.arch + str(args.depth)

if args.remove_layers !=0:
    input = torch.randn(1, 3, 32, 32)
    omacs, oparams = profile(model, inputs=(input, ))
    prune_cifar_resnet56(model)
    macs, params = profile(model, inputs=(input, ))
    print('flops reduction %0.3f'%((1-(macs/omacs))*100))
    #print(model)

if args.cuda:
    model.cuda(device=args.gpuid)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(device=args.gpuid), target.cuda(args.gpuid)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(device=args.gpuid), target.cuda(args.gpuid)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'best_model.pth.tar'))

best_prec1 = 0.
prec1 = test()
if args.action == 'train':
    for epoch in range(args.start_epoch, args.epochs):
        lr = args.lr * (args.lr_decay_scalar ** (epoch // args.lr_decay_every))
        if lr != args.lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        train(epoch)
        prec1 = test()
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'cfg': model.cfg
        }, is_best, filepath=args.save)

    print('Best achieved: ',best_prec1)

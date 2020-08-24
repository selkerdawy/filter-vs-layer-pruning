from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cifar as models
import pdb
import pickle
from utils import Bar, TensorboardLogger, Logger, AverageMeter, accuracy, mkdir_p, savefig
from pprint import pformat


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--data', default='path to dataset if imagenet', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--pretrained', type=str, help='path to pretrained model')
parser.add_argument('--preimprinted', default=None, type=str,
                    help='path to allweights (default: none)')

# Imprint
parser.add_argument('--action', type=str,default='train', help='visualize or imprint')
parser.add_argument('--ratio', type=float, default=1.0, help='Ratio of dataset to be used in imprinting calculation')

parser.add_argument('--criterion', default='', type=str, help='Path to criterion')
parser.add_argument('--remove-layers', default=0, type=int, help='How many layers/blocks to remove')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg19_bn)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
#Device options

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset

if os.path.isfile(args.checkpoint):
    args.tbdir = os.path.join(os.path.dirname(args.checkpoint),'tb')
else:
    args.tbdir = os.path.join(args.checkpoint,'tb')

if os.path.exists(args.tbdir):
    print('Removing old tb folder ..')
    shutil.rmtree(args.tbdir)

tb = TensorboardLogger(args.tbdir, is_master=True)

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print('Seed: %d'%args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

tb.writer.add_text("args", pformat(vars(args)))
activations_hooks = {}
activations_GP = {}

#CIFAR
target = 512
layer_fmap_size = {}
per_layer_sensitivity = {}
do_norm_embedding = False
do_norm_weights = False



class ZeroOut(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x*0


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
        blockid = sortedidx[j]# - 1
        for whichlayer, g in enumerate(groupscum):
            if g > blockid:
                break

        whichblock = blockid%groups[whichlayer]
        if (whichblock > 0 or whichlayer ==0 ) and blockid >= 0 :
            block = mapping[whichlayer][whichblock]
            print('       Removing block %d from group %d'%(whichblock,whichlayer+1))
            mapping[whichlayer][whichblock].conv1 = nn.Identity()
            mapping[whichlayer][whichblock].bn1 = nn.Identity()
            mapping[whichlayer][whichblock].relu = nn.Identity()
            mapping[whichlayer][whichblock].conv2 = nn.Identity()
            mapping[whichlayer][whichblock].bn2 = ZeroOut()
            mapping[whichlayer][whichblock].pruned = True
            i+=1
        j+=1

def prune_cifar_resnet56(model):
    import pickle
    crit = pickle.load(open(args.criterion, 'rb'))
    depth = 56#args.depth
    groups = [(depth - 2) // 6]*3
    get_pruned_resnet56(model, crit, groups)


def normalize_01(arr):
   shifted = arr + abs(arr.min())
   shifted /= shifted.max()
   return shifted

def l2_norm(input, normalize=do_norm_embedding):
    input_size = input.size()
    GP = input
    nbatches = input.size(0)

    if input.dim() > 2 and input_size[-1] > 1:
        GP = input.norm(p=2, dim=(2,3))
        # Calculate the h/w of the features so that after flatten, all layers have nearest 'target' features
        dim = round(math.sqrt(target/input_size[1]))
        input = nn.AdaptiveAvgPool2d(dim)(input).view(input.size(0),-1)
        input_size = input.size()

    input_ = input.view(nbatches,-1)

    if normalize:
        norm = input_.norm(p=2, dim=1)
        _output = torch.div(input_, norm.view(-1, 1).expand_as(input_))
    else:
        _output = input

    output = _output.view(nbatches,-1)

    return output, GP

def gather_results(fmaps, ngpus):
    for k, v in fmaps.items():
        tmp = []
        # Reorder
        for gpu in range(ngpus):
            tmp+=[v[gpu]]

        # Concatenate
        fmaps[k] = torch.cat(tmp)
    return fmaps

def collect_activations(model, dataloader, nbatches=500, num_classes=1000):

    activations_hooks = {}
    activations_GP = {}
    hooks_location = []

    def hook(self, input, output):
        norm_output, GP = l2_norm(output)

        key = self.key
        gpu = output.device.index
        activations_hooks[key][gpu] = norm_output.data.cpu().detach()
        activations_GP[key][gpu] = GP.data.cpu().detach()
        layer_fmap_size[key] = output.shape[1:]

    lst = list(model.modules())
    idx = 0
    while idx < len(lst):

        m = lst[idx]
        if 'BasicBlock' in str(type(m)):# or (hasattr(m,'pruned') and 'Identity' in str(type(m))):
            m.key = '%03d'%idx
            print('Adding hook %s for '%m.key, m)
            hooks_location += [idx]
            activations_hooks[m.key] = {}
            activations_GP[m.key] = {}
            m.register_forward_hook(hook)

        '''
        # Add hooks to all conv and FC except last fc, probably not the best way to check for last FC --> TODO
        if 'Conv' in str(type(m)) or 'Linear' in str(type(m)) and m.weight.shape[0]!=num_classes:
            print('Adding hook for ', m)
            for i in range(idx+1,len(lst)):
                cur = str(type(lst[i]))
                if 'Conv' in cur or 'Linear' in cur or 'Sequential' in cur or 'Bottleneck' in cur or 'BasicBlock' in cur or 'Pool' in cur:
                    idx = i-1
                    break

            m = lst[idx]
            hooks_location += [idx]
            m.key = '%03d'%idx
            activations_hooks[m.key] = {}
            activations_GP[m.key] = {}
            m.register_forward_hook(hook)
        '''

        idx += 1

    # Each layer is a key with value as a map. This map's key is class id and value is the avg feature map for that class on that layer.
    activations_per_layer_per_class = {}
    activations_gp_per_layer_per_class = {}
    nclass = {}
    ndata = len(dataloader)
    ngpus = torch.cuda.device_count()

    print('Collecting embeddings ...')
    for batch_idx ,(img, lbls) in enumerate(dataloader):
        if batch_idx >= nbatches:
            break
        model(img)

        activations_hooks = gather_results(activations_hooks, ngpus)
        activations_GP = gather_results(activations_GP, ngpus)

        for (k, v), (_, gp_v) in zip(activations_hooks.items(), activations_GP.items()):
            if k not in activations_per_layer_per_class:
                activations_per_layer_per_class[k] = {}
                activations_gp_per_layer_per_class[k] = {}

            for i, lbl in enumerate(lbls):
                lbl = lbl.item()
                if lbl in activations_per_layer_per_class[k]:
                    activations_per_layer_per_class[k][lbl] += v[i]
                    activations_gp_per_layer_per_class[k][lbl] += gp_v[i]
                    nclass[lbl] += 1
                else:
                    activations_per_layer_per_class[k][lbl] = v[i]
                    activations_gp_per_layer_per_class[k][lbl] = gp_v[i]
                    nclass[lbl] = 1

            #activations_hooks[k] = torch.zeros_like(v)

        activations_hooks = {k:{} for k,v in activations_hooks.items()}
        activations_GP = {k:{} for k,v in activations_GP.items()}

        if batch_idx %10 == 0:
            print(batch_idx,'/',nbatches)

    return activations_per_layer_per_class, activations_gp_per_layer_per_class, nclass, hooks_location

def apply_branch_classifier(allweights, model, hooks_location, num_classes = 1000):

    import threading

    lock = threading.Lock()

    def hook_classifier(self, input, output):
        global activations_hooks

        norm_output, _ = l2_norm(output)
        key = self.key
        gpu = output.device.index

        with lock:
            if key not in activations_hooks:
                activations_hooks[key] = {}

        dotproduct = torch.mm(norm_output, allweights[key].t().to(norm_output.device))
        activations_hooks[key][gpu] = dotproduct.clone()
        return

    lst = list(model.modules())

    print('Adding proxy classifiers ...')
    for idx, m in enumerate(lst):
        if idx in hooks_location:
            m.key = '%03d'%idx
            m.register_forward_hook(hook_classifier)

def get_cifar_loader(dataloader):

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    return trainloader, testloader


def bar_plot(accs, savepath, tickname='block', saveplot = True):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.style.use('ggplot')
    #plt.style.use('bmh')
    importance = [a for _,a in accs.items()]
    n = np.sqrt(sum(np.asarray(importance)**2))
    importance = np.asarray(importance)/max(importance)
    fig, ax = plt.subplots()
    sortedidx = np.argsort(importance)
    for i, idx in enumerate(sortedidx):
        plt.bar(idx, importance[idx], label='%s%d'%(tickname,idx))

    lbls = ['%s%d'%(tickname,i) for i in range(len(importance))]
    plt.title('%s-%s'%(args.arch, args.remove_layers))
    plt.ylabel('Normalized importance')
    plt.xticks(np.arange(0, len(importance), 1.0), lbls, rotation=70)
    plt.legend(prop={"size":9})
    width = 8
    height = 7

    fig.set_size_inches(width, height)
    if saveplot:
        plt.savefig(savepath)

def plot_vgg(accs):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.style.use('seaborn')

    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.15, bottom=.12, right=.99, top=.97)

    acc = [v for k,v in accs.items()]
    acc = [ round(a) for a in acc]
    lbls = ['conv%02d'%(i+1) for i in range(len(acc))]
    lbls[-1] = 'GT'
    w = 6
    bw = 5
    x = [i*w for i,_ in enumerate(acc)]
    plt.bar(x,acc, bw)

    for i, v in enumerate(acc):
        c = 'blue'
        if i >=10 and i<=14:
            c = 'red'
        plt.text(i*w - bw/1.7, v + .4, str(int(round(v))), color=c)#, fontsize=14)

    ax.set_ylabel('Accuracy', size='large')
    plt.xticks(range(0,len(acc)*w,w), lbls, rotation='vertical')
    ax.set_xticklabels(lbls)

    #plt.show()

def main():

    global target

    input_size = 32
    print("==> creating dataloader for '{}'".format(args.dataset))
    if args.dataset.lower() == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        trainloader, testloader = get_cifar_loader(dataloader)
    elif args.dataset.lower() == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
        trainloader, testloader = get_cifar_loader(dataloader)

    # Model
    print("==> creating model '{}'".format(args.arch))

    if args.arch.startswith('vgg'):
        model = models.__dict__[args.arch](num_classes=num_classes, input_size=input_size)
    if args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](num_classes=num_classes, input_size=input_size, dataset=args.dataset)
        global target
        target = 2048
    else:
        print('Not supported network')
        raise NotImplementedError

    checkpoint = torch.load(args.pretrained)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    # Just in case model saved wrapped with DataParallel
    checkpoint = {k.replace('module.',''): v for k,v in  checkpoint.items()}

    if args.remove_layers !=0:
        prune_cifar_resnet56(model)

    model.load_state_dict(checkpoint, strict=False)

    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    #print(model)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1e6))

    losses = []
    criterion = nn.CrossEntropyLoss()
    losses.append(criterion)
    l1_crit = nn.L1Loss(size_average=False)
    losses.append(l1_crit)

    nbatches = int(args.ratio * len(trainloader))
    if args.preimprinted is None:
        nbatches = int(args.ratio * len(trainloader))
        activations_per_layer_per_class, activations_gp_per_layer_per_class, nclass, hooks_location = collect_activations(model, trainloader, nbatches = nbatches, num_classes=num_classes)

        for k,v in activations_per_layer_per_class.items():
            for cls, fmap in v.items():
                activations_per_layer_per_class[k][cls] = fmap/nclass[cls]

        for k,v in activations_gp_per_layer_per_class.items():
            for cls, fmap in v.items():
                tmp = fmap/nclass[cls]
                if do_norm_weights:
                    tmp /= tmp.norm(p=2)
                activations_gp_per_layer_per_class[k][cls] = tmp

        # Calculate weights per layer
        allweights = {}
        for k,v in activations_per_layer_per_class.items():
            nclass = len(v.keys())
            nfmap = len(v[0])
            new_weight = torch.zeros(num_classes, nfmap)
            assert nclass == num_classes, "Increase ratio of dataset, not all classes are sampled"
            for cls, fmap in v.items():
                new_weight[cls] = fmap / fmap.norm(p=2)
            allweights[k] = new_weight
        '''
        savepath = os.path.join(os.path.dirname(args.tbdir), 'imprinted-weights.pkl')
        with open(savepath, 'wb') as f:
            global layer_fmap_size
            pickle.dump([layer_fmap_size, hooks_location, allweights], f)
        '''

    else:
        layer_fmap_size, hooks_location, allweights = pickle.load(open(args.preimprinted,'rb'))
        print('Loaded weights from ', args.preimprinted)
        #asd = prune_by_weights(allweights)

    apply_branch_classifier(allweights, model, hooks_location, num_classes = num_classes)

    accs = test(testloader, model, len(allweights), use_cuda)
    savepath = os.path.join(os.path.dirname(args.tbdir), 'accs%d.pkl'%args.remove_layers)
    pickle.dump(accs, open(savepath,'wb'))
    importance = [a for _, a in accs.items()]
    crit = [cur - prev for prev, cur in zip([0] + importance, importance)]
    pickle.dump(crit, open(savepath.replace('accs','crit'),'wb'))
    #bar_plot(accs, savepath.replace('.pkl','.png'))

def test(testloader, model,nlayers, use_cuda):
    global activations_hooks

    top1 = [0]*(nlayers+1)
    top5 = [0]*(nlayers+1)
    for i in range(nlayers+1):
        top1[i] = AverageMeter()
        top5[i] = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    ngpus = torch.cuda.device_count()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time

        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
        #inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        #print(targets)
        outputs = model(inputs)

        activations_hooks = gather_results(activations_hooks, ngpus)

        strtop = ""
        # measure accuracy at each layer
        #activations_hooks={}
        activations_hooks['last'] = outputs
        ninput = inputs.size(0)
        accs = {}
        final_dist = nn.LogSoftmax(dim=1)(outputs).cpu().detach().numpy()
        sorted_layers = sorted(activations_hooks.keys())

        for idx, k in enumerate(sorted_layers):
            v = activations_hooks[k]
            prec1, prec5 = accuracy(v.data, targets.data, topk=(1, 5))
            top1[idx].update(prec1.item(), ninput)
            top5[idx].update(prec5.item(), ninput)
            #print(idx, prec1.item(), v.shape, top1[idx].count, ninput)
            accs["%02d"%idx] = top1[idx].avg
            strtop += "top1_{:d}: {:.4f} | ".format(idx, top1[idx].avg)


        activations_hooks = {}
        # measure elapsed time
        end = time.time()

        #tb.writer.add_scalars("Accuracy_per_layer", accs,global_step = batch_idx)
        print("({batch}/{size}) {top:s}".format(batch=batch_idx + 1, size=len(testloader), top=strtop))

    bar.finish()
    return accs

if __name__ == '__main__':
    main()


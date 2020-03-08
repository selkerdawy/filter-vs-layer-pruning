import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import pdb

def bar_plot(importance, title, tickname, saveplot=False, show=True):

    plt.style.use('ggplot')
    #plt.style.use('bmh')
    n = np.sqrt(sum(np.asarray(importance)**2))
    importance = np.asarray(importance)/max(importance)
    fig, ax = plt.subplots()
    sortedidx = np.argsort(importance)
    for i, idx in enumerate(sortedidx):
        plt.bar(idx, importance[idx], label='%s%d'%(tickname,idx))

    lbls = ['%s%d'%(tickname,i) for i in range(len(importance))]
    plt.title(title)
    plt.ylabel('Normalized importance')
    plt.xticks(np.arange(0, len(importance), 1.0), lbls, rotation=70)
    plt.legend(prop={"size":9})
    width = 8
    height = 7

    fig.set_size_inches(width, height)
    if saveplot:
        plt.savefig(title.replace(' ','_'))
    if show:
        plt.show()


def get_ensemble_ranking(criteria_per_method):
    layer_ranks = np.asarray([0]*len(criteria_per_method[list(criteria_per_method.keys())[0]]))
    for k,v in criteria_per_method.items():
        idx_ranking = np.argsort(v)
        layer_ranks += idx_ranking

    return layer_ranks

def get_resnet50_importance(method, criteria):
    group = [3,4,6,3]
    i = 4 if method == 22 else 0
    importance = []
    jmb = 2
    for b, g in enumerate(group):
        for j in range(g):
            downsample = 0.
            if j == 0 and method == 22:
                downsample = (criteria[b].mean())
            if downsample == 0.:
                importance += [np.mean([a.mean() for a in criteria[i:i+jmb]])]
            else:
                importance += [np.mean([a.mean() for a in criteria[i:i+jmb]] + [downsample])]

            i += jmb
            print(importance[-1])

    return importance

def plot(nw, dataset):

    mapping = {0:'weight taylor', 2: 'weight magnitude', 6: 'feature map taylor', 22: 'gate taylor', 30: 'BN scale', 31: 'BN taylor'}
    nw_mapping = {'resnet50':'ResNet56', 'vgg11': 'VGG11_BN'}
    xaxislbl = 'block' if 'resnet' in nw else 'layer'
    per_method_importance = {}

    for method in mapping.keys():
        #method=0

        filename = '%s/%s/one_shot_criterion%d/criteria_%d_final.pickle'%(dataset, nw, method , method)

        criteria = pickle.load(open(filename, 'rb'))
        print(filename)

        if nw == 'resnet50' and dataset == 'ImageNet':
            importance = get_resnet50_importance(method, criteria)
        else:
            importance = [a.mean() for a in criteria]

        per_method_importance[method] = importance

        bar_plot(importance, '%s %s block importance using %s'%(dataset, nw_mapping[nw], mapping[method]), xaxislbl, saveplot=True, show=False)

    ensemble = get_ensemble_ranking(per_method_importance)

    bar_plot(importance, '%s %s block importance using ensemble'%(dataset,nw_mapping[nw]), xaxislbl, saveplot=True, show=False)

if __name__ == '__main__':
    plot(sys.argv[1], sys.argv[2])

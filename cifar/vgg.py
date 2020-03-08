import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import math
import pdb
import sys
sys.path.append("../..")

__all__ = [
    'VGG', 'vgg19_bn',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):

    def __init__(self, features_masks, classifier, **kwargs):
        super(VGG, self).__init__()
        self.features = features_masks
        self.classifier = classifier
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_classifier(config, final_conv):
    layers = []
    last_input = final_conv * 7 * 7
    dp=0.5
    for v in config:
        layers += [nn.Linear(last_input, v), nn.Dropout(p=dp), nn.ReLU(True)] #DO added to match number of layers in pretrained
        last_input = v

    layers += [nn.Linear(last_input, num_classes)]
    return nn.Sequential(*layers)


def make_layers(cfg, batch_norm=False, weight='uniform', **kwargs):
    layers = []
    in_channels = 3
    xshape = kwargs['input_size']
    num_classes = kwargs['num_classes']
    cfg = cfg
    final_conv = -1

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            xshape /= 2
        else:
            final_conv = v
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    if num_classes == 1000:
        c_cfg = kwargs['c_cfg'] #cfg[-2:]
        layers += [nn.AdaptiveAvgPool2d(7)]
        classifier = make_classifier(c_cfg, final_conv)
    else:
        classifier = nn.Linear(final_conv, num_classes)

    features = nn.Sequential(*layers)
    return features, classifier

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'default': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg19_bn(arch_dict = None,**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""

    defaults  = {'num_classes':1000, 'input_size':244, 'c_cfg': None}
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = v

    if arch_dict is None:
        arch_dict = cfg['E']
        if kwargs['num_classes'] == 1000:
            kwargs['c_cfg'] = [4096,4096]
    print(arch_dict, kwargs['c_cfg'])

    features, classifier = make_layers(arch_dict, batch_norm=True, **kwargs)
    model = VGG(features, classifier)
    return model


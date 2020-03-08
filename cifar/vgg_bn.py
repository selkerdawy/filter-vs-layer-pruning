# based on https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from utils.layers import GateLayer

__all__ = ["VGG", "vgg11_bn"]

class VGG(torch.nn.Module):

    def __init__(self, features, classifier,
                 num_classes=1000, init_weights=True, dropout=False, classifier_BN=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = classifier

        if num_classes == 1000:
            self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        else:
            self.avgpool = torch.nn.Identity()
            self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(512, num_classes)
            )

        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


def make_classifier_linear_BN(num_classes, config, final_conv):
    layers = []
    layers += [nn.Linear(final_conv * 7 * 7, config[0]), nn.BatchNorm1d(config[0]), GateLayer(config[0], config[0], [1, -1]), nn.ReLU(True)]
    layers += [nn.Linear(config[0], config[1]), nn.BatchNorm1d(config[1]),  GateLayer(config[1], config[1], [1, -1]), nn.ReLU(True)]
    layers += [nn.Linear(config[1], num_classes)]
    return nn.Sequential(*layers)

def make_layers(cfg, batch_norm=False, classifier_BN = True):
    layers = []
    in_channels = 3
    final_conv = 3
    for v in cfg:
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v), GateLayer(v, v, [1, -1, 1, 1]), torch.nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, GateLayer(v, v, [1, -1, 1, 1]), torch.nn.ReLU(inplace=True)]
            final_conv = v
            in_channels = v

    if classifier_BN:
        classifier = make_classifier_linear_BN(1000, [4096, 4096], final_conv)
    else:
        classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096), # 0> 0
            GateLayer(4096, 4096, [1, -1]),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.5 if dropout else 0.0),
            torch.nn.Linear(4096, 4096), # 3 > 4
            GateLayer(4096, 4096, [1, -1]),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.5 if dropout else 0.0),
            torch.nn.Linear(4096, num_classes), # 6 > 8
        )

    return torch.nn.Sequential(*layers), classifier


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    features, classifier = make_layers(cfgs[cfg], batch_norm=batch_norm)
    model = VGG(features, classifier , **kwargs)
    return model

def vgg19_bn(pretrained='', progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (str): If path is not empty, load state_dict --> CIFAR only
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
    if len(pretrained) > 0:
        state_dict = torch.load(pretrained)['state_dict']
        state_dict = {k.replace('module.',''):v for k, v in state_dict.items()}
        #print(state_dict.keys())
        #print(model.state_dict().keys())
        sofar = 0
        newstate ={}
        for k,v in state_dict.items():
            if 'classifier' in k:
                break
            parts = k.split('.')
            newname = '.'.join([parts[0], str(int(parts[1]) + sofar), parts[2]])
            newstate[newname] = v
            #print(k,' to ', newname, ' sofar ', sofar)
            if 'running_var' in k:
                sofar+=1

        newstate['classifier.0.weight'] = state_dict['classifier.weight']
        newstate['classifier.0.bias'] = state_dict['classifier.bias']
        #print(newstate.keys())
        model.load_state_dict(newstate, strict=False)

    return model

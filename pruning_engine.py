
from __future__ import print_function
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

import numpy as np
import pdb
import math

from copy import deepcopy
import itertools
import pickle
import json
from collections import OrderedDict, defaultdict

from tqdm import tqdm

METHOD_ENCODING = {0: "Taylor_weight", 1: "Random", 2: "Weight norm", 3: "Weight_abs",
                   6: "Taylor_output", 10: "OBD", 11: "Taylor_gate_SO",
                   22: "Taylor_gate", 23: "Taylor_gate_FG", 30: "BN_weight", 31: "BN_Taylor",
                   60: "Layerwise Relevance Propagation",
                   61: "Method 22 with normalized grad by adam"}

# Code is based on
# https://github.com/NVlabs/Taylor_pruning

# Method is encoded as an integer that mapping is shown above.
# Methods map to the paper as follows:
# 0 - Taylor_weight - Conv weight/conv/linear weight with Taylor FO In Table 2 and Table 1
# 1 - Random        - Random
# 2 - Weight norm   - Weight magnitude/ weight
# 3 - Weight_abs    - Not used
# 6 - Taylor_output - Taylor-output as is [27]
# 10- OBD           - OBD
# 11- Taylor_gate_SO- Taylor SO
# 22- Taylor_gate   - Gate after BN in Table 2, Taylor FO in Table 1
# 23- Taylor_gate_FG- uses gradient per example to compute Taylor FO, Taylor FO- FG in Table 1, Gate after BN - FG in Table 2
# 30- BN_weight     - BN scale in Table 2
# 31- BN_Taylor     - BN scale Taylor FO in Table 2
# 60- Layerwise Revelance Propagation   - Heatmapping examples for filter importance

class PruningConfigReader(object):
    def __init__(self):
        self.pruning_settings = {}
        self.config = None

    def read_config(self, filename):
        # reads .json file and sets values as pruning_settings for pruning

        with open(filename, "r") as f:
            config = json.load(f)

        self.config = config

        self.read_field_value("method", 0)
        self.read_field_value("frequency", 500)
        self.read_field_value("prune_per_iteration", 2)
        self.read_field_value("maximum_pruning_iterations", 10000)
        self.read_field_value("starting_neuron", 0)

        self.read_field_value("fixed_layer", -1)
        # self.read_field_value("use_momentum", False)

        self.read_field_value("pruning_threshold", 100)
        self.read_field_value("start_pruning_after_n_iterations", 0)
        # self.read_field_value("use_momentum", False)
        self.read_field_value("do_iterative_pruning", True)
        self.read_field_value("fixed_criteria", False)
        self.read_field_value("seed", 0)
        self.read_field_value("pruning_momentum", 0.9)
        self.read_field_value("flops_regularization", 0.0)
        self.read_field_value("prune_neurons_max", 1)

        self.read_field_value("group_size", 1)
        self.read_field_value("prune_latency_ratio", -1)
        self.read_field_value("layer_minimum_neurons", 1)
        self.read_field_value("prune_loss_batch_patience", -1)
        self.read_field_value("prune_loss_batch_patience_window_size", 1)

    def read_field_value(self, key, default):
        param = default
        if key in self.config:
            param = self.config[key]

        self.pruning_settings[key] = param

    def get_parameters(self):
        return self.pruning_settings


class pytorch_pruning(object):
    def __init__(self, parameters, pruning_settings=dict(), log_folder=None):
        def initialize_parameter(object_name, settings, key, def_value):
            '''
            Function check if key is in the settings and sets it, otherwise puts default momentum
            :param object_name: reference to the object instance
            :param settings: dict of settings
            :param def_value: def value for the parameter to be putted into the field if it doesn't work
            :return:
            void
            '''
            value = def_value
            if key in settings.keys():
                value = settings[key]
            setattr(object_name, key, value)

        # store some statistics
        self.min_criteria_value = 1e6
        self.max_criteria_value = 0.0
        self.median_criteria_value = 0.0
        self.neuron_units = 0
        self.all_neuron_units = 0
        self.pruned_neurons = 0
        self.gradient_norm_final = 0.0
        self.flops_regularization = 0.0 #not used in the paper
        self.pruning_iterations_done = 0
        self.full_model_latency = 0
        self.estimated_model_latency = 0

        # initialize_parameter(self, pruning_settings, 'use_momentum', False)
        initialize_parameter(self, pruning_settings, 'pruning_momentum', 0.9)
        initialize_parameter(self, pruning_settings, 'flops_regularization', 0.0)
        self.momentum_coeff = self.pruning_momentum
        self.use_momentum = self.pruning_momentum > 0.0

        initialize_parameter(self, pruning_settings, 'prune_per_iteration', 1)
        initialize_parameter(self, pruning_settings, 'start_pruning_after_n_iterations', 0)
        initialize_parameter(self, pruning_settings, 'prune_neurons_max', 0)
        initialize_parameter(self, pruning_settings, 'maximum_pruning_iterations', 0)
        initialize_parameter(self, pruning_settings, 'pruning_silent', False)
        initialize_parameter(self, pruning_settings, 'l2_normalization_per_layer', False)
        initialize_parameter(self, pruning_settings, 'fixed_criteria', False)
        initialize_parameter(self, pruning_settings, 'starting_neuron', 0)
        initialize_parameter(self, pruning_settings, 'frequency', 30)
        initialize_parameter(self, pruning_settings, 'pruning_threshold', 100)
        initialize_parameter(self, pruning_settings, 'fixed_layer', -1)
        initialize_parameter(self, pruning_settings, 'combination_ID', 0)
        initialize_parameter(self, pruning_settings, 'seed', 0)
        initialize_parameter(self, pruning_settings, 'group_size', 1)

        initialize_parameter(self, pruning_settings, 'method', 0)
        initialize_parameter(self, pruning_settings, 'prune_latency_ratio', -1)
        initialize_parameter(self, pruning_settings, "layer_minimum_neurons", 1)
        initialize_parameter(self, pruning_settings, "prune_loss_batch_patience", -1)
        initialize_parameter(self, pruning_settings, "prune_loss_batch_patience_window_size", 1)

        # Hessian related parameters
        self.temp_hessian = [] # list to store Hessian
        self.hessian_first_time = True

        self.parameters = list()

        ##get pruning parameters
        for parameter in parameters:
            parameter_value = parameter["parameter"]
            self.parameters.append(parameter_value)

        if self.fixed_layer == -1:
            ##prune all layers
            self.prune_layers = [True for parameter in self.parameters]
        else:
            ##prune only one layer
            self.prune_layers = [False, ]*len(self.parameters)
            self.prune_layers[self.fixed_layer] = True

        self.iterations_done = 0

        self.prune_network_criteria = list()
        self.prune_network_accomulate = {"by_layer": list(), "averaged": list(), "averaged_cpu": list()}

        self.pruning_gates = list()
        for layer in range(len(self.parameters)):
            self.prune_network_criteria.append(list())

            for key in self.prune_network_accomulate.keys():
                self.prune_network_accomulate[key].append(list())

            self.pruning_gates.append(np.ones(len(self.parameters[layer]),))
            layer_now_criteria = self.prune_network_criteria[-1]
            for unit in range(len(self.parameters[layer])):
                layer_now_criteria.append(0.0)

        # logging setup
        self.log_folder = log_folder
        self.folder_to_write_debug = self.log_folder + '/debug/'
        if not os.path.exists(self.folder_to_write_debug):
            os.makedirs(self.folder_to_write_debug)

        self.method_25_first_done = True

        if self.method == 40 or self.method == 50 or self.method == 25:
            self.oracle_dict = {"layer_pruning": -1, "initial_loss": 0.0, "loss_list": list(), "neuron": list(), "iterations": 0}
            self.method_25_first_done = False

        if self.method == 25:
            with open("./utils/study/oracle.pickle","rb") as f:
                oracle_list = pickle.load(f)

            self.oracle_dict["loss_list"] = oracle_list

        self.needs_hessian = False
        if self.method in [10, 11]:
            self.needs_hessian = True

        # useful for storing data of the experiment
        self.data_logger = dict()
        self.data_logger["pruning_neurons"] = list()
        self.data_logger["pruning_accuracy"] = list()
        self.data_logger["pruning_loss"] = list()
        self.data_logger["method"] = self.method
        self.data_logger["prune_per_iteration"] = self.prune_per_iteration
        self.data_logger["combination_ID"] = list()
        self.data_logger["fixed_layer"] = self.fixed_layer
        self.data_logger["frequency"] = self.frequency
        self.data_logger["starting_neuron"] = self.starting_neuron
        self.data_logger["use_momentum"] = self.use_momentum

        self.data_logger["time_stamp"] = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

        if hasattr(self, 'seed'):
            self.data_logger["seed"] = self.seed

        self.data_logger["filename"] = "%s/data_logger_seed_%d_%s.p"%(log_folder, self.data_logger["seed"], self.data_logger["time_stamp"])
        if self.method == 50:
            self.data_logger["filename"] = "%s/data_logger_seed_%d_neuron_%d_%s.p"%(log_folder, self.starting_neuron, self.data_logger["seed"], self.data_logger["time_stamp"])
        self.log_folder = log_folder

        # the rest of initializations
        self.pruned_neurons = self.starting_neuron

        self.util_loss_tracker = 0.0
        self.util_acc_tracker = 0.0
        self.util_loss_tracker_num = 0.0

        self.loss_tracker_exp = ExpMeter()
        # stores results of the pruning, 0 - unsuccessful, 1 - successful
        self.res_pruning = 0

        self.iter_step = -1

        self.train_writer = None

        self.set_moment_zero = True
        self.pruning_mask_from = ""

        self.prune_loss_batch_patience_counter = 0
        self.prune_loss_batch_patience_value = None
        self.prune_loss_batch_patience_window = []
        self.prune_loss_batch_patience_window_avg = None

        if self.method == 60:
            # trying for multiple gpu/device support
            self.method_60_activations = defaultdict(list)
            self.method_60_targets = None

    def add_criteria(self, optimizer):
        '''
        This method adds criteria to global list given batch stats.
        '''

        if self.fixed_criteria:
            if self.pruning_iterations_done > self.start_pruning_after_n_iterations :
                return 0

        method_60_required = False

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            nunits = self.parameters[layer].size(0)
            eps = 1e-8

            if len(self.pruning_mask_from) > 0:
                # preload pruning mask
                self.method = -1
                criteria_for_layer = torch.from_numpy(self.loaded_mask_criteria[layer]).type(torch.FloatTensor).cuda(nonblocking=True)

            if self.method == 0:
                # First order Taylor expansion on the weight
                criteria_for_layer = (self.parameters[layer]*self.parameters[layer].grad ).data.pow(2).view(nunits,-1).sum(dim=1)
            elif self.method == 1:
                # random pruning
                criteria_for_layer = np.random.uniform(low=0, high=5, size=(nunits,))
            elif self.method == 2:
                # min weight
                criteria_for_layer = self.parameters[layer].pow(2).view(nunits,-1).sum(dim=1).data
            elif self.method == 3:
                # weight_abs
                criteria_for_layer = self.parameters[layer].abs().view(nunits,-1).sum(dim=1).data
            elif self.method == 63:
                # min weight based on next layer, tested only on alexnet
                if layer == len(self.prune_layers)-1:
                    last_layer = list(self.model_instance.modules())[-1]
                    criteria_for_layer = last_layer.weight.pow(2).transpose(0,1).contiguous().view(nunits,-1)
                else:
                    criteria_for_layer = self.parameters[layer+1].pow(2).transpose(0,1).contiguous().view(nunits,-1)
                criteria_for_layer = criteria_for_layer.sum(dim=1).data
            elif self.method == 6:
                # ICLR2017 Taylor on output of the layer
                if 1:
                    criteria_for_layer = self.parameters[layer].full_grad_iclr2017
                    #criteria_for_layer = criteria_for_layer / (np.linalg.norm(criteria_for_layer) + eps)
                    criteria_for_layer = criteria_for_layer / (np.linalg.norm(criteria_for_layer.cpu().numpy()) + eps)
            elif self.method == 10:
                # diagonal of Hessian
                criteria_for_layer = (self.parameters[layer] * torch.diag(self.temp_hessian[layer])).data.view(nunits,
                                                                                                               -1).sum(
                    dim=1)
            elif self.method == 11:
                #  second order Taylor expansion for loss change in the network
                criteria_for_layer = (-(self.parameters[layer] * self.parameters[layer].grad).data + 0.5 * (
                            self.parameters[layer] * self.parameters[layer] * torch.diag(
                        self.temp_hessian[layer])).data).pow(2)

            elif self.method == 22:
                # Taylor pruning on gate
                criteria_for_layer = (self.parameters[layer]*self.parameters[layer].grad).data.pow(2).view(nunits, -1).sum(dim=1)
                if hasattr(self, "dataset"):
                    # fix for skip connection pruning, gradient will be accumulated instead of being averaged
                    if self.dataset == "Imagenet":
                        if hasattr(self, "model"):
                            if not ("noskip" in self.model):
                                if "resnet" in self.model:
                                    mult = 3.0
                                    if layer == 1:
                                        mult = 4.0
                                    elif layer == 2:
                                        mult = 23.0 if "resnet101" in self.model else mult
                                        mult = 6.0  if "resnet34" in self.model else mult
                                        mult = 6.0  if "resnet50" in self.model else mult

                                    criteria_for_layer /= mult

            elif self.method == 61:
                # Taylor pruning with grad normalized by adam
                grad = self.parameters[layer].grad
                state=optimizer.state[self.parameters[layer]]

                if 'exp_avg' not in state:
                    raise ValueError("Normalized grad method (61) is only supported with adam")

                beta1, beta2 = (0.9, 0.999)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)

                step_size = 1. / bias_correction1
                grad = step_size * (exp_avg/denom)

                criteria_for_layer = (self.parameters[layer]*grad).data.pow(2).view(nunits, -1).sum(dim=1)
                if hasattr(self, "dataset"):
                    # fix for skip connection pruning, gradient will be accumulated instead of being averaged
                    if self.dataset == "Imagenet":
                        if hasattr(self, "model"):
                            if not ("noskip" in self.model):
                                if "resnet" in self.model:
                                    mult = 3.0
                                    if layer == 1:
                                        mult = 4.0
                                    elif layer == 2:
                                        mult = 23.0 if "resnet101" in self.model else mult
                                        mult = 6.0  if "resnet34" in self.model else mult
                                        mult = 6.0  if "resnet50" in self.model else mult

                                    criteria_for_layer /= mult

            elif self.method == 23:
                # Taylor pruning on gate with computing full gradient
                criteria_for_layer = (self.parameters[layer].full_grad.t()).data.pow(2).view(nunits,-1).sum(dim=1)

            elif self.method == 30:
                # batch normalization based pruning
                # by scale (weight) of the batchnorm
                criteria_for_layer = (self.parameters[layer]).data.abs().view(nunits, -1).sum(dim=1)

            elif self.method == 31:
                # Taylor FO on BN
                if hasattr(self.parameters[layer], "bias"):
                    criteria_for_layer = (self.parameters[layer]*self.parameters[layer].grad +
                                          self.parameters[layer].bias*self.parameters[layer].bias.grad ).data.pow(2).view(nunits,-1).sum(dim=1)
                else:
                    criteria_for_layer = (
                                self.parameters[layer] * self.parameters[layer].grad).data.pow(2).view(nunits, -1).sum(dim=1)

            elif self.method == 40:
                # ORACLE on the fly that reevaluates itslef every pruning step
                criteria_for_layer = np.asarray(self.oracle_dict["loss_list"][layer]).copy()
                self.oracle_dict["loss_list"][layer] = list()
            elif self.method == 50:
                # combinatorial pruning - evaluates all possibilities of removing N neurons
                criteria_for_layer = np.asarray(self.oracle_dict["loss_list"][layer]).copy()
                self.oracle_dict["loss_list"][layer] = list()
            elif self.method == 60:
                method_60_required = True
                break
            else:
                pass

            if self.iterations_done == 0:
                self.prune_network_accomulate["by_layer"][layer] = criteria_for_layer
            else:
                self.prune_network_accomulate["by_layer"][layer] += criteria_for_layer

        if self.method == 60:
            if self.res_pruning == -1:
                # this is expensive, don't calculate it if res_pruning is finished
                pass
            else:
                assert method_60_required
                assert self.dataset == "Imagenet", "only works for imagenet"

                # These contain activations from multiple GPUs with split batch size
                # while T and R are for one GPU only. Either all of these calculations need to be done per GPU
                # batch or we need to combine all of the activations into one GPU device and hope
                # everything fits into vram...

                all_T = torch.nn.functional.one_hot(self.method_60_targets, 1000)
                all_scores = None
                keep_idxs = []
                target_offset = 0
                with tqdm(total = sum([len(v) - 1 for v in self.method_60_activations.values()]), desc="LRP (Method 60)") as pbar:
                    for device in self.method_60_activations.keys():
                        pbar.set_postfix({"device": device})
                        A = self.method_60_activations[device]
                        batch_size = A[len(A) - 1][1].size(0) # get batch size
                        T = all_T[target_offset: target_offset + batch_size].to(device)
                        R = None
                        scores = None
                        output = None
                        for l in range(1, len(A))[::-1]:
                            layer, inp, layer_output = A[l]
                            pbar.set_postfix({"device": device, "layer_id": l})
                            if output is None:
                                output = layer_output
                                R = [None] * len(A) + [(output * T).data]
                                scores = [None] * len(A)
                            inp = inp.data.requires_grad_(True)
                            # print(device, l, type(layer), inp.shape)
                            if isinstance(layer, torch.nn.MaxPool2d):
                                layer = torch.nn.AvgPool2d(
                                    layer.kernel_size, stride=layer.stride, padding=layer.padding)
                            if isinstance(layer, torch.nn.Conv2d) or \
                                isinstance(layer, torch.nn.AvgPool2d) or \
                                isinstance(layer, torch.nn.AdaptiveAvgPool2d):

                                # All these are hyperparameters and require manual, painful tuning; assumes VGG11BN!
                                if l <= 3:
                                    rho = lambda p: p + 0.25 * p.clamp(min=0)
                                    incr = lambda z: z + 1e-9
                                if 4 <= l <= 6:
                                    rho = lambda p: p
                                    incr = lambda z: z + 1e-9 * ((z**2).mean()**.5).data
                                if l >= 7:
                                    rho = lambda p:p
                                    incr = lambda z: z + 1e-9

                                n_layer = newlayer(layer, rho)

                                # Cuda out of memory error, perform these calculations without batching
                                for batch_idx in range(0, inp.size(0)):
                                    batch_inp = inp[batch_idx].unsqueeze(0)
                                    # Step 1
                                    z = incr(n_layer.forward(batch_inp))
                                    if type(layer) == torch.nn.AdaptiveAvgPool2d:
                                        z = z.view(z.shape[0], -1, 1, 1)
                                    # enforce all zeros are some small epsilon value
                                    z[z==0] = 1e-9
                                    # Step 2
                                    s = (R[l+1][batch_idx]/z).data
                                    # Step 3
                                    (z*s).sum().backward()
                                    c = inp.grad[batch_idx].unsqueeze(0)
                                    # Step 4
                                    if R[l] is None:
                                        R[l] = [None] * batch_size
                                    R[l][batch_idx] = (batch_inp * c).data

                                    try:
                                        assert torch.isnan(c).sum() == 0
                                    except Exception as e:
                                        import pdb; pdb.set_trace()
                                        print(e)

                                    # use C for filter importance score
                                    if scores[l] is None:
                                        scores[l] = c
                                    else:
                                        scores[l] += c

                            else:
                                if R[l] is None:
                                    R[l] = R[l+1]
                                else:
                                    R[l] += R[l+1]
                            # only keep the indicies of modules that we care about
                            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                                if l not in keep_idxs:
                                    keep_idxs.append(l)
                            pbar.update()
                        self.method_60_activations[device] = self.method_60_activations[device][len(A):]
                        target_offset += batch_size

                        # average scores over batch size
                        try:
                            # check non NaNs
                            for score_idx, s in enumerate(scores):
                                if s is not None:
                                    assert torch.isnan(s).sum() == 0
                                    scores[score_idx] = (s/batch_size).cpu()
                        except Exception as e:
                            import pdb; pdb.set_trace()
                            print(e)

                        if all_scores is None:
                            all_scores = scores
                        else:
                            for score_idx, s in enumerate(all_scores):
                                if s is not None:
                                    all_scores[score_idx] = s + scores[score_idx]

                assert all_scores is not None
                assert len(keep_idxs) == len(self.prune_network_accomulate["by_layer"])
                keep_idxs.sort()

                # Clamp all scores such that they are between -1 and 1, then scale by number of neurons
                layerwise_neurons = self._count_layerwise_number_of_neurons()

                assert len(layerwise_neurons) == len(keep_idxs)
                total_number_of_unpruned_filters = sum([total_unpruned_filters for (total_filters, total_unpruned_filters) in layerwise_neurons.values()])

                for layer, keep_idx in enumerate(keep_idxs):
                    # resize the all_scores to match parameter layer shape
                    score_shape = self.parameters[layer].shape[0]
                    criteria_for_layer = all_scores[keep_idx].view(score_shape, -1).sum(dim=1)

                    try:
                        # check shape matches
                        assert criteria_for_layer.shape == self.parameters[layer].shape
                        assert criteria_for_layer.shape[0] == layerwise_neurons[layer][0]
                        # check non NaNs
                        assert torch.isnan(criteria_for_layer).sum() == 0
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()

                    # current layer normalization multiplier
                    mult = layerwise_neurons[layer][1] / total_number_of_unpruned_filters

                    # clamp to -1 and 1!
                    old_min = criteria_for_layer.min()
                    old_max = criteria_for_layer.max()
                    old_range = old_max - old_min
                    new_min = -1
                    new_max = 1
                    new_range = new_max - new_min
                    for idx, criteria_value in enumerate(criteria_for_layer):
                        updated_criteria_value = (((criteria_value - old_min) * new_range) / old_range) + new_min
                        # normalize by multiplying with number of unpruned filters of total number of unpruned filters
                        # criteria_for_layer[idx] = updated_criteria_value * mult

                        criteria_for_layer[idx] = updated_criteria_value # no mult

                    if self.iterations_done == 0:
                        self.prune_network_accomulate["by_layer"][layer] = criteria_for_layer
                    else:
                        self.prune_network_accomulate["by_layer"][layer] += criteria_for_layer

        self.iterations_done += 1

    @staticmethod
    def group_criteria(list_criteria_per_layer, group_size=1):
        '''
        Function combine criteria per neuron into groups of size group_size.
        Output is a list of groups organized by layers. Length of output is a number of layers.
        The criterion for the group is computed as an average of member's criteria.
        Input:
        list_criteria_per_layer - list of criteria per neuron organized per layer
        group_size - number of neurons per group

        Output:
        groups - groups organized per layer. Each group element is a tuple of 2: (index of neurons, criterion)
        '''
        groups = list()

        for layer in list_criteria_per_layer:
            layer_groups = list()
            indeces = np.argsort(layer)
            for group_id in range(int(np.ceil(len(layer)/group_size))):
                current_group = slice(group_id*group_size, min((group_id+1)*group_size, len(layer)))
                values = [layer[ind] for ind in indeces[current_group]]
                group = [indeces[current_group], sum(values)]

                layer_groups.append(group)
            groups.append(layer_groups)

        return groups

    def compute_saliency(self):
        '''
        Method performs pruning based on precomputed criteria values. Needs to run after add_criteria()
        '''
        def write_to_debug(what_write_name, what_write_value):
            # Aux function to store information in the text file
            with open(self.log_debug, 'a') as f:
                f.write("{} {}\n".format(what_write_name,what_write_value))

        def nothing(what_write_name, what_write_value):
            pass

        if self.method == 50:
            write_to_debug = nothing

        # compute loss since the last pruning and decide if to prune:
        if self.util_loss_tracker_num > 0:
            validation_error = self.util_loss_tracker / self.util_loss_tracker_num
            validation_error_long = validation_error
            acc = self.util_acc_tracker / self.util_loss_tracker_num
        else:
            print(
                "compute loss and run self.util_add_loss(loss.item()) before running this")
            validation_error = 0.0
            acc = 0.0

        self.util_training_loss = validation_error
        self.util_training_acc = acc

        # reset training loss tracker
        self.util_loss_tracker = 0.0
        self.util_acc_tracker = 0.0
        self.util_loss_tracker_num = 0

        if validation_error > self.pruning_threshold:
            ## if error is big then skip pruning
            print("skipping pruning", validation_error, "(%f)"%validation_error_long, self.pruning_threshold)
            if self.method != 4:
                self.res_pruning = -1
                return -1

        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # if reached max number of pruning iterations -> exit
            if self.res_pruning != -1:
                print("maximum number of prune iterations reached, skipping pruning")
            self.res_pruning = -1
            return -1

        if self.prune_neurons_max != -1 and self.prune_neurons_max <= (self.all_neuron_units - self.neuron_units):
            # if reached max number of neurons to prune -> exit
            if self.res_pruning != -1:
                print("target number of pruned neurons reached, skipping pruning")
            self.res_pruning = -1
            if self.method == 60:
                # trying for multiple gpu/device support
                activation_keys = list(self.method_60_activations.keys())
                for key in activation_keys:
                    del self.method_60_activations[key]
                self.method_60_targets = None
                torch.cuda.empty_cache()
            return -1

        if self.prune_latency_ratio != -1 and self.full_model_latency != 0 and self.estimated_model_latency != 0:
            # check latency ratio reached
            target_latency = self.full_model_latency * self.prune_latency_ratio
            if self.estimated_model_latency <= target_latency:
                if self.res_pruning != -1:
                    print("target latency reached, skipping pruning")
                self.res_pruning = -1
                return -1

        self.full_list_of_criteria = list()

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            if self.iterations_done > 0:
                # momentum turned to be useless and even reduces performance
                contribution = self.prune_network_accomulate["by_layer"][layer] / self.iterations_done
                if self.pruning_iterations_done == 0 or not self.use_momentum or (self.method in [4, 40, 50]):
                    self.prune_network_accomulate["averaged"][layer] = contribution
                else:
                    # use momentum to accumulate criteria over several pruning iterations:
                    self.prune_network_accomulate["averaged"][layer] = self.momentum_coeff*self.prune_network_accomulate["averaged"][layer]+(1.0- self.momentum_coeff)*contribution

                current_layer = self.prune_network_accomulate["averaged"][layer]
                if not (self.method in [1, 4, 40, 15, 50]):
                    current_layer = current_layer.cpu().numpy()

                if self.l2_normalization_per_layer:
                    eps = 1e-8
                    current_layer = current_layer / (np.linalg.norm(current_layer) + eps)

                self.prune_network_accomulate["averaged_cpu"][layer] = current_layer
            else:
                print("First do some add_criteria iterations")
                exit()

            for unit in range(len(current_layer)):
                criterion_now = current_layer[unit]

                # make sure that pruned neurons have 0 criteria
                self.prune_network_criteria[layer][unit] =  criterion_now * self.pruning_gates[layer][unit]

                if self.method == 50:
                    self.prune_network_criteria[layer][unit] =  criterion_now

        # count number of neurons
        all_neuron_units, neuron_units = self._count_number_of_neurons()
        self.neuron_units = neuron_units
        self.all_neuron_units = all_neuron_units

        # store criteria_result into file
        if not self.pruning_silent:
            import pickle
            store_criteria = self.prune_network_accomulate["averaged_cpu"]
            pickle.dump(store_criteria, open(self.folder_to_write_debug + "criteria_%04d.pickle"%self.pruning_iterations_done, "wb"))
            if self.pruning_iterations_done == 0:
                pickle.dump(store_criteria, open(self.log_folder + "criteria_%d.pickle"%self.method, "wb"))
            pickle.dump(store_criteria, open(self.log_folder + "criteria_%d_final.pickle"%self.method, "wb"))

        if not self.fixed_criteria:
            self.iterations_done = 0

        # create groups per layer
        groups = self.group_criteria(self.prune_network_criteria, group_size=self.group_size)

        # apply flops regularization
        # if self.flops_regularization > 0.0:
        #     self.apply_flops_regularization(groups, mu=self.flops_regularization)

        # enforce minimum neurons per layer
        for layer_idx, layer in enumerate(groups):
            # ensure the most important min number of neurons per layer are never pruned
            layer_criterias = [(group_criteria, group_idx) for (group_idx, (_, group_criteria)) in enumerate(layer)]
            layer_criterias.sort(key=lambda tup: tup[0], reverse=True)
            for _, group_idx in layer_criterias[:self.layer_minimum_neurons]:
                groups[layer_idx][group_idx][1] = float('inf')

        layerwise_num_neurons = self._count_layerwise_number_of_neurons()
        self.layerwise_neurons = layerwise_num_neurons
        # if lookup table, apply layerwise contribution to overall latency
        if hasattr(self, 'lut'):
            network_def = get_network_def_from_model(self.model_instance, self.input_shape)
            layerwise_latency, net_def_partial = compute_layerwise_latency_from_groups_and_lut(self.model, network_def, self.lut, self.full_rbf, groups, layerwise_num_neurons)

            if not self.only_estimate_latency:
                # update groups score using latency values
                for layer_idx, layer in enumerate(groups):
                    for group_idx, (group_out_channel, group_criteria) in enumerate(layer):
                        lat_group_out_channel, group_latency = layerwise_latency[layer_idx][group_idx]
                        assert lat_group_out_channel == group_out_channel
                        if group_latency != 0:
                            groups[layer_idx][group_idx][1] = group_criteria / group_latency
                        else:
                            groups[layer_idx][group_idx][1] = group_criteria

            # update network def for new latency computation
            for k, v in net_def_partial.items():
                (new_in_channels, new_out_channels) = v
                network_def[k][KEY_NUM_IN_CHANNELS] = new_in_channels
                network_def[k][KEY_NUM_OUT_CHANNELS] = new_out_channels
            if self.model == "mobilenet":
                network_def["fc"][KEY_NUM_IN_CHANNELS] = new_out_channels

            self.estimated_model_latency = compute_latency_from_lookup_table(network_def, self.lut, self.full_rbf)
            if self.full_model_latency == 0:
                self.full_model_latency = self.estimated_model_latency
                self.target_latency = self.full_model_latency * self.prune_latency_ratio
            print("estimated_latency: %.3e, target latency: %.3e" %(self.estimated_model_latency, self.target_latency))

        if hasattr(self, 'bilinear'):
            if not self.only_estimate_latency:
                raise ValueError("Layer wise latency ranking not supported with bilinear model.")

            self.estimated_model_latency = self._compute_latency_from_bilinear_model(layerwise_num_neurons)
            if self.full_model_latency == 0:
                self.full_model_latency = self.estimated_model_latency
                self.target_latency = self.full_model_latency * self.prune_latency_ratio
            print("estimated_latency: %.3e, target latency: %.3e" %(self.estimated_model_latency, self.target_latency))


        # get an array of all criteria from groups
        all_criteria = np.asarray([group[1] for layer in groups for group in layer]).reshape(-1)

        prune_neurons_now = (self.pruned_neurons + self.prune_per_iteration)//self.group_size - 1
        if self.prune_neurons_max != -1:
            prune_neurons_now = min(len(all_criteria)-1, min(prune_neurons_now, self.prune_neurons_max//self.group_size - 1))

        # adaptively estimate threshold given a number of neurons to be removed
        threshold_now = np.sort(all_criteria)[prune_neurons_now]

        if self.method == 50:
            # combinatorial approach
            threshold_now = 0.5
            self.pruning_iterations_done = self.combination_ID
            self.data_logger["combination_ID"].append(self.combination_ID-1)
            self.combination_ID += 1
            self.reset_oracle_pruning()
            print("full_combinatorial: combination ", self.combination_ID)

        self.pruning_iterations_done += 1

        self.log_debug = self.folder_to_write_debug + 'debugOutput_pruning_%08d' % (
            self.pruning_iterations_done) + '.txt'
        write_to_debug("method", self.method)
        write_to_debug("pruned_neurons", self.pruned_neurons)
        write_to_debug("pruning_iterations_done", self.pruning_iterations_done)
        write_to_debug("neuron_units", neuron_units)
        write_to_debug("all_neuron_units", all_neuron_units)
        write_to_debug("threshold_now", threshold_now)
        write_to_debug("groups_total", sum([len(layer) for layer in groups]))
        if self.estimated_model_latency is not None:
            write_to_debug("estimated_latency", self.estimated_model_latency)

        if self.prune_latency_ratio != -1 and self.full_model_latency != 0 and self.estimated_model_latency != 0:
            # check latency ratio reached
            target_latency = self.full_model_latency * self.prune_latency_ratio
            if self.estimated_model_latency <= target_latency:
                if self.res_pruning != -1:
                    print("target latency reached, skipping pruning")
                self.res_pruning = -1
                return -1

        if self.pruning_iterations_done < self.start_pruning_after_n_iterations:
            self.res_pruning = -1
            return -1

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            write_to_debug("\nLayer:", layer)
            write_to_debug("units:", len(self.parameters[layer]))

            if self.prune_per_iteration == 0:
                continue

            for group in groups[layer]:
                if group[1] <= threshold_now:
                    for unit in group[0]:
                        # do actual pruning
                        self.pruning_gates[layer][unit] *= 0.0
                        self.parameters[layer].data[unit] *= 0.0

            write_to_debug("pruned_perc:", [np.nonzero(1.0-self.pruning_gates[layer])[0].size, len(self.parameters[layer])])

        # count number of neurons
        all_neuron_units, neuron_units = self._count_number_of_neurons()
        self.neuron_units = neuron_units
        self.all_neuron_units = all_neuron_units
        self.pruned_neurons = all_neuron_units-neuron_units

        if self.method == 25:
            self.method_25_first_done = True

        self.threshold_now = threshold_now
        try:
            # criteria values should ignore infinite as that's a predefined constraint
            non_inf_criteria = all_criteria[all_criteria < float('inf')]
            good_bounds_criteria = non_inf_criteria[non_inf_criteria > 0.0]
            self.min_criteria_value = good_bounds_criteria.min()
            self.max_criteria_value = good_bounds_criteria.max()
            self.median_criteria_value = np.median(good_bounds_criteria)
        except:
            self.min_criteria_value = 0.0
            self.max_criteria_value = 0.0
            self.median_criteria_value = 0.0

        # set result to successful
        self.res_pruning = 1

    def _get_layer_utilization_bilinear(self, idx, prev, cur):

        #Scales start with number of input channels
        scales = self.bilinear['preprocessor.scales'][idx:idx+2]
        prev /= scales[0]
        cur /= scales[1]

        weight = self.bilinear['regressor.0.weight'][idx]

        res = weight * prev * cur

        return res


    def _compute_latency_from_bilinear_model(self, layerwise_num_neurons):
        '''
        Function computes total latency using self.bilinear for self.model based on layerwise_num_neurons
        :return:
        Estimated latency
        '''

        res = 0.0
        prev = 3
        for idx, (_,cur) in layerwise_num_neurons.items():
            res += self._get_layer_utilization_bilinear(idx, prev, cur)
            prev = cur

        # add classes * last layer params
        weight_class = self.bilinear['regressor.0.weight'][-1]
        scale_class = self.bilinear['preprocessor.scales'][-2]
        res += weight_class * cur/scale_class * 1 #normalized numclass/numclass
        res += self.bilinear['regressor.0.bias'].item() #bias

        return res


    def _count_number_of_neurons(self):
        '''
        Function computes number of total neurons and number of active neurons
        :return:
        all_neuron_units - number of neurons considered for pruning
        neuron_units     - number of not pruned neurons in the model
        '''
        all_neuron_units = 0
        neuron_units = 0
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            all_neuron_units += len( self.parameters[layer] )
            for unit in range(len( self.parameters[layer] )):
                if len(self.parameters[layer].data.size()) > 1:
                    statistics = self.parameters[layer].data[unit].abs().sum()
                else:
                    statistics = self.parameters[layer].data[unit]

                if statistics > 0.0:
                    neuron_units += 1

        return all_neuron_units, neuron_units

    def _count_layerwise_number_of_neurons(self):
        '''
        Function computes number of total neurons and number of active neurons
        :return:
        all_neuron_units - number of neurons considered for pruning
        neuron_units     - number of not pruned neurons in the model
        '''
        layerwise_neurons = OrderedDict()
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            all_neuron_units = len( self.parameters[layer] )
            neuron_units = 0
            for unit in range(len( self.parameters[layer] )):
                if len(self.parameters[layer].data.size()) > 1:
                    statistics = self.parameters[layer].data[unit].abs().sum()
                else:
                    statistics = self.parameters[layer].data[unit]

                if statistics > 0.0:
                    neuron_units += 1

            layerwise_neurons[layer] = (all_neuron_units, neuron_units)
        return layerwise_neurons


    def set_weights_oracle_pruning(self):
        '''
        sets gates/weights to zero to evaluate pruning
        will reuse weights for pruning
        only for oracle pruning
        '''

        for layer,if_prune in enumerate(self.prune_layers_oracle):
            if not if_prune:
                continue

            if self.method == 40:
                self.parameters[layer].data = deepcopy(torch.from_numpy(self.stored_weights).cuda())

            for unit in range(len(self.parameters[layer])):
                if self.method == 40:
                    self.pruning_gates[layer][unit] = 1.0

                    if unit == self.oracle_unit:
                        self.pruning_gates[layer][unit] *= 0.0
                        self.parameters[layer].data[unit] *= 0.0

                        # if 'momentum_buffer' in optimizer.state[self.parameters[layer]].keys():
                        #     optimizer.state[self.parameters[layer]]['momentum_buffer'][unit] *= 0.0
        return 1

    def reset_oracle_pruning(self):
        '''
        Method restores weights to original after masking for Oracle pruning
        :return:
        '''
        for layer, if_prune in enumerate(self.prune_layers_oracle):
            if not if_prune:
                continue

            if self.method == 40 or self.method == 50:
                self.parameters[layer].data = deepcopy(torch.from_numpy(self.stored_weights).cuda())

            for unit in range(len( self.parameters[layer])):
                if self.method == 40 or self.method == 50:
                    self.pruning_gates[layer][unit] = 1.0

    def enforce_pruning(self):
        '''
        Method sets parameters ang gates to 0 for pruned neurons.
        Helpful if optimizer will change weights from being zero (due to regularization etc.)
        '''
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            for unit in range(len(self.parameters[layer])):
                if self.pruning_gates[layer][unit] == 0.0:
                    self.parameters[layer].data[unit] *= 0.0

    def compute_hessian(self, loss):
        '''
        Computes Hessian per layer of the loss with respect to self.parameters, currently implemented only for gates
        '''

        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # if reached max number of pruning iterations -> exit
            self.res_pruning = -1
            return -1

        self.temp_hessian = list()
        for layer_indx, parameter in enumerate(self.parameters):
            # print("Computing Hessian current/total layers:",layer_indx,"/",len(self.parameters))
            if self.prune_layers[layer_indx]:
                grad_params = torch.autograd.grad(loss, parameter, create_graph=True)
                length_grad = len(grad_params[0])
                hessian = torch.zeros(length_grad, length_grad)

                cnt = 0
                for parameter_loc in range(len(parameter)):
                    if parameter[parameter_loc].data.cpu().numpy().sum() == 0.0:
                        continue

                    grad_params2 = torch.autograd.grad(grad_params[0][parameter_loc], parameter, create_graph=True)
                    hessian[parameter_loc, :] = grad_params2[0].data

            else:
                length_grad = len(parameter)
                hessian = torch.zeros(length_grad, length_grad)

            self.temp_hessian.append(torch.FloatTensor(hessian.cpu().numpy()).cuda())

    def run_full_oracle(self, model, data, target, criterion, initial_loss):
        '''
        Runs oracle on all data by setting to 0 every neuron and running forward pass
        '''

        # stop adding data if needed
        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # if reached max number of pruning iterations -> exit
            self.res_pruning = -1
            return -1

        if self.method == 40:
            # for oracle let's try to do the best possible oracle by evaluating all neurons for each batch
            self.oracle_dict["initial_loss"] += initial_loss
            self.oracle_dict["iterations"]   += 1

            # import pdb; pdb.set_trace()
            if hasattr(self, 'stored_pruning'):
                if self.stored_pruning['use_now']:
                    # load first list of criteria
                    print("use previous computed priors")
                    for layer_index, layer_parameters in enumerate(self.parameters):

                        # start list of estiamtes for the layer if it is empty
                        if len(self.oracle_dict["loss_list"]) < layer_index + 1:
                            self.oracle_dict["loss_list"].append(list())

                        if self.prune_layers[layer_index] == False:
                            continue

                        self.oracle_dict["loss_list"][layer_index] = self.stored_pruning['criteria_start'][layer_index]
                    self.pruned_neurons = self.stored_pruning['neuron_start']
                    return 1

            # do first pass with precomputed values
            for layer_index, layer_parameters in enumerate(self.parameters):
                # start list of estimates for the layer if it is empty
                if len(self.oracle_dict["loss_list"]) < layer_index + 1:
                    self.oracle_dict["loss_list"].append(list())

                if not self.prune_layers[layer_index]:
                    continue
                # copy original prune_layer variable that sets layers to be prunned
                self.prune_layers_oracle = [False, ]*len(self.parameters)
                self.prune_layers_oracle[layer_index] = True
                # store weights for future to recover
                self.stored_weights = deepcopy(self.parameters[layer_index].data.cpu().numpy())

                for neurion_id, neuron in enumerate(layer_parameters):
                    # set neuron to zero
                    self.oracle_unit = neurion_id
                    self.set_weights_oracle_pruning()

                    if self.stored_weights[neurion_id].sum() == 0.0:
                        new_loss = initial_loss
                    else:
                        outputs = model(data)
                        loss = criterion(outputs, target)
                        new_loss = loss.item()

                    # define loss
                    oracle_value = abs(initial_loss - new_loss)
                    # relative loss for testing:
                    # oracle_value = initial_loss - new_loss

                    if len(self.oracle_dict["loss_list"][layer_index]) == 0:
                        self.oracle_dict["loss_list"][layer_index] = [oracle_value, ]
                    elif len(self.oracle_dict["loss_list"][layer_index]) < neurion_id+1:
                        self.oracle_dict["loss_list"][layer_index].append(oracle_value)
                    else:
                        self.oracle_dict["loss_list"][layer_index][neurion_id] += oracle_value

                self.reset_oracle_pruning()

        elif self.method == 50:
            if self.pruning_iterations_done == 0:
                # store weights again
                self.stored_weights = deepcopy(self.parameters[self.fixed_layer].data.cpu().numpy())

            self.set_next_combination()

        else:
            pass
            # print("Full oracle only works with the methods: {}".format(40))

    def report_loss_neuron(self, training_loss, training_acc, train_writer = None, neurons_left = 0):
        '''
        method to store stistics during pruning to the log file
        :param training_loss:
        :param training_acc:
        :param train_writer:
        :param neurons_left:
        :return:
        void
        '''
        if train_writer is not None:
            train_writer.add_scalar('loss_neuron', training_loss, self.all_neuron_units-self.neuron_units)

        self.data_logger["pruning_neurons"].append(self.all_neuron_units-self.neuron_units)
        self.data_logger["pruning_loss"].append(training_loss)
        self.data_logger["pruning_accuracy"].append(training_acc)

        self.write_log_file()

    def report_criteria(self, per_layer_criteria_avg, train_writer):

        for layer, criteria in enumerate(per_layer_criteria_avg):
            train_writer.add_histogram('criteria_%03d'%layer, criteria.cpu().detach().type(torch.float32), global_step = self.pruning_iterations_done)

    def report_signature(self, layerwise_neurons, training_acc, train_writer = None):
        '''
        method to store stistics during pruning to tb
        :param layerwise_neurons:
        :param training_acc:
        :param train_writer:
        :return:
        void
        '''
        if train_writer is not None:
            signature = {}
            for idx, (_, cur) in layerwise_neurons.items():
                signature['layer_%d'%idx] = cur
            train_writer.add_scalars('filters', signature, global_step = self.pruning_iterations_done)
        print('Filters per layer: ')
        print(signature)


    def write_log_file(self):
        with open(self.data_logger["filename"], "wb") as f:
            pickle.dump(self.data_logger, f)

    def load_mask(self):
        '''Method loads precomputed criteria for pruning
        :return:
        '''
        if not len(self.pruning_mask_from)>0:
            print("pruning_engine.load_mask(): did not find mask file, will load nothing")
        else:
            if not os.path.isfile(self.pruning_mask_from):
                print("pruning_engine.load_mask(): file doesn't exist", self.pruning_mask_from)
                print("pruning_engine.load_mask(): check it, exit,", self.pruning_mask_from)
                exit()

            with open(self.pruning_mask_from, 'rb') as f:
                self.loaded_mask_criteria = pickle.load(f)

            print("pruning_engine.load_mask(): loaded criteria from", self.pruning_mask_from)

    def set_next_combination(self):
        '''
        For combinatorial pruning only
        '''
        if self.method == 50:

            self.oracle_dict["iterations"]   += 1

            for layer_index, layer_parameters in enumerate(self.parameters):

                ##start list of estiamtes for the layer if it is empty
                if len(self.oracle_dict["loss_list"]) < layer_index + 1:
                    self.oracle_dict["loss_list"].append(list())

                if self.prune_layers[layer_index] == False:
                    continue

                nunits = len(layer_parameters)

                comb_num = -1
                found_combination = False
                for it in itertools.combinations(range(nunits), self.starting_neuron):
                    comb_num += 1
                    if comb_num == int(self.combination_ID):
                        found_combination = True
                        break

                # import pdb; pdb.set_trace()
                if not found_combination:
                    print("didn't find needed combination, exit")
                    exit()

                self.prune_layers_oracle = self.prune_layers.copy()
                self.prune_layers_oracle = [False,]*len(self.parameters)
                self.prune_layers_oracle[layer_index] = True

                criteria_for_layer = np.ones((nunits,))
                criteria_for_layer[list(it)] = 0.0

                if len(self.oracle_dict["loss_list"][layer_index]) == 0:
                    self.oracle_dict["loss_list"][layer_index] = criteria_for_layer
                else:
                    self.oracle_dict["loss_list"][layer_index] += criteria_for_layer

    def report_to_tensorboard(self, train_writer, processed_batches):
        '''
        Log data with tensorboard
        '''
        gradient_norm_final_before = self.gradient_norm_final
        train_writer.add_scalar('Neurons_left', self.neuron_units, processed_batches)
        train_writer.add_scalar('Criteria_min', self.min_criteria_value, self.pruning_iterations_done)
        train_writer.add_scalar('Criteria_max', self.max_criteria_value, self.pruning_iterations_done)
        train_writer.add_scalar('Criteria_median', self.median_criteria_value, self.pruning_iterations_done)
        train_writer.add_scalar('Gradient_norm_before', gradient_norm_final_before, self.pruning_iterations_done)
        train_writer.add_scalar('Pruning_threshold', self.threshold_now, self.pruning_iterations_done)
        train_writer.add_scalar('Estimated_latency', self.estimated_model_latency, self.pruning_iterations_done)

    def util_add_loss(self, training_loss_current, training_acc):
        # keeps track of current loss
        self.util_loss_tracker += training_loss_current
        self.util_acc_tracker  += training_acc
        self.util_loss_tracker_num += 1
        self.loss_tracker_exp.update(training_loss_current)
        # self.acc_tracker_exp.update(training_acc)

    def do_step(self, loss=None, optimizer=None, neurons_left=0, training_acc=0.0, epoch=-1, batch_idx=-1, num_batches=-1, losses_tracker=None):
        '''
        do one step of pruning,
        1) Add importance estimate
        2) checks if loss is above threshold
        3) performs one step of pruning if needed
        '''
        self.iter_step += 1
        niter = self.iter_step

        # # sets pruned weights to zero
        # self.enforce_pruning()

        # stop if pruned maximum amount
        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # exit if we pruned enough
            self.res_pruning = -1
            return -1

        # sets pruned weights to zero
        self.enforce_pruning()

        # compute criteria for given batch
        self.add_criteria(optimizer)

        # small script to keep track of training loss since the last pruning
        self.util_add_loss(loss, training_acc)

        if self.frequency > 0:
            if ((niter-1) % self.frequency == 0) and (niter != 0) and (self.res_pruning==1):
                self.report_loss_neuron(
                    self.util_training_loss,
                    training_acc=self.util_training_acc,
                    train_writer=self.train_writer,
                    neurons_left=neurons_left)
                self.report_signature(
                    self.layerwise_neurons,
                    training_acc=self.util_training_acc,
                    train_writer=self.train_writer)
                self.report_criteria(
                        self.prune_network_accomulate["averaged"],
                        self.train_writer)

            if niter % self.frequency == 0 and niter != 0:
                # do actual pruning, output: 1 - good, 0 - no pruning

                self.compute_saliency()
                self.set_optimizer_prune_zero(optimizer=optimizer)

                training_loss = self.util_training_loss
                if self.res_pruning == 1:
                    print(
                        "Pruning: Units", self.neuron_units, "/", self.all_neuron_units,
                        "loss", training_loss, "Zeroed", self.pruned_neurons,
                        "criteria min:{}/max:{:2.7f}".format(self.min_criteria_value, self.max_criteria_value))
        elif self.res_pruning != -1:
            assert self.prune_loss_batch_patience > 0, "Must set prune frequency or prune_loss_batch_patience"
            assert losses_tracker != None, "losses_tracker must have average loss"

            # use prune loss batch patience window for pruning criteria
            if len(self.prune_loss_batch_patience_window) == 0:
                self.prune_loss_batch_patience_counter = 0
                self.prune_loss_batch_patience_window.append(losses_tracker.val)
                print(
                    "Epoch: [{0}][{1}/{2}]\tINIT prune_loss_batch_patience_window = {bpv}".format(
                        epoch, batch_idx, num_batches, bpv=losses_tracker.val))
            elif len(self.prune_loss_batch_patience_window) < self.prune_loss_batch_patience_window_size:
                self.prune_loss_batch_patience_window.append(losses_tracker.val)
            elif len(self.prune_loss_batch_patience_window) == self.prune_loss_batch_patience_window_size:
                self.prune_loss_batch_patience_window = self.prune_loss_batch_patience_window[1:] + [losses_tracker.val]

            if len(self.prune_loss_batch_patience_window) == self.prune_loss_batch_patience_window_size:
                window_avg = np.average(self.prune_loss_batch_patience_window)

                if self.prune_loss_batch_patience_window_avg is None:
                    self.prune_loss_batch_patience_window_avg = window_avg
                    print("Epoch: [{0}][{1}/{2}]\tWindow Full: INIT prune_loss_batch_patience_window_avg = {win_avg}".format(
                            epoch, batch_idx, num_batches, win_avg=window_avg))
                    window_avg_delta = None
                else:
                    window_avg_delta = np.abs(window_avg - self.prune_loss_batch_patience_window_avg)
                    self.prune_loss_batch_patience_window_avg = window_avg

                if window_avg_delta is None:
                    pass
                elif self.prune_loss_batch_patience_value is None:
                    self.prune_loss_batch_patience_value = window_avg_delta
                elif self.prune_loss_batch_patience_value < window_avg_delta:
                    self.prune_loss_batch_patience_counter += 1
                else:
                    print("Epoch: [{0}][{1}/{2}]\tUPDATE prune_loss_batch_patience_value = {bpv}, RESET counter from {cnt} to 0".format(
                        epoch, batch_idx, num_batches, bpv=window_avg_delta,
                        cnt=self.prune_loss_batch_patience_counter))
                    self.prune_loss_batch_patience_counter = 0
                    self.prune_loss_batch_patience_value = window_avg_delta

            if self.prune_loss_batch_patience_counter >= self.prune_loss_batch_patience:
                print("Epoch: [{0}][{1}/{2}]\tcounter ({cnt}) exceeded prune_loss_batch_patience, do step prune".format(
                    epoch, batch_idx, num_batches, cnt=self.prune_loss_batch_patience_counter))
                self.compute_saliency()
                self.set_optimizer_prune_zero(optimizer=optimizer)
                if self.res_pruning == 1:
                    self.report_loss_neuron(
                        self.util_training_loss,
                        training_acc=self.util_training_acc,
                        train_writer=self.train_writer,
                        neurons_left=neurons_left)
                    print(
                        "Pruning: Units", self.neuron_units, "/", self.all_neuron_units,
                        "loss", self.util_training_loss, "Zeroed", self.pruned_neurons,
                        "criteria min:{}/max:{:2.7f}".format(self.min_criteria_value, self.max_criteria_value))
                self.prune_loss_batch_patience_counter = 0
                self.prune_loss_batch_patience_value = None
                self.prune_loss_batch_patience_window = []
                self.prune_loss_batch_patience_window_avg = None


    def set_optimizer_prune_zero(self, optimizer=None):
        '''
        Method sets momentum buffer /step / exp_avg / exp_avg_sq to zero for pruned neurons.
        :return:
        void
        '''
        for layer in range(len(self.pruning_gates)):
            if not self.prune_layers[layer]:
                continue
            for unit in range(len(self.pruning_gates[layer])):
                if not self.pruning_gates[layer][unit]:
                    continue
                if 'momentum_buffer' in optimizer.state[self.parameters[layer]].keys():
                    optimizer.state[self.parameters[layer]]['momentum_buffer'][unit] *= 0.0
                if 'step' in optimizer.state[self.parameters[layer]].keys():
                    optimizer.state[self.parameters[layer]]['step'] *= 0.0
                if 'exp_avg' in optimizer.state[self.parameters[layer]].keys():
                    optimizer.state[self.parameters[layer]]['exp_avg'][unit].zero_()
                if 'exp_avg_sq' in optimizer.state[self.parameters[layer]].keys():
                    optimizer.state[self.parameters[layer]]['exp_avg_sq'][unit].zero_()

    def connect_tensorboard(self, tensorboard):
        '''
        Function connects tensorboard to pruning engine
        '''
        self.tensorboard = True
        self.train_writer = tensorboard

    def update_flops(self, stats=None):
        '''
        Function updates flops for potential regularization
        :param stats: a list of flops per parameter
        :return:
        '''
        self.per_layer_flops = list()
        if len(stats["flops"]) < 1:
            return -1
        for pruning_param in self.gates_to_params:
            if isinstance(pruning_param, list):
                # parameter spans many blocks, will aggregate over them
                self.per_layer_flops.append(sum([stats['flops'][a] for a in pruning_param]))
            else:
                self.per_layer_flops.append(stats['flops'][pruning_param])

    def apply_flops_regularization(self, groups, mu=0.1):
        '''
        Function applieregularisation to computed importance per layer
        :param groups: a list of groups organized per layer
        :param mu: regularization coefficient
        :return:
        '''
        if len(self.per_layer_flops) < 1:
            return -1

        for layer_id, layer in enumerate(groups):
            for group in layer:
                # import pdb; pdb.set_trace()
                total_neurons = len(group[0])
                group[1] = group[1] - mu*(self.per_layer_flops[layer_id]*total_neurons)


def prepare_pruning_list(pruning_settings, model, model_name, pruning_mask_from='', name=''):
    '''
    Function returns a list of parameters from model to be considered for pruning.
    Depending on the pruning method and strategy different parameters are selected (conv kernels, BN parameters etc)
    :param pruning_settings:
    :param model:
    :return:
    '''
    # Function creates a list of layer that will be pruned based o user selection

    ADD_BY_GATES = True  # gates add artificially they have weight == 1 and not trained, but gradient is important. see models/lenet.py
    ADD_BY_WEIGHTS = ADD_BY_BN = False

    pruning_method = pruning_settings['method']

    pruning_parameters_list = list()
    if ADD_BY_GATES:

        first_step = True
        prev_module = None
        prev_module2 = None
        last_gate = None
        print("network structure")
        for module_indx, m in enumerate(model.modules()):
            # print(module_indx, m)
            if hasattr(m, "do_not_update"):
                m_to_add = m

                if (pruning_method != 23) and (pruning_method != 6):
                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}
                else:
                    def just_hook(self, grad_input, grad_output):
                        # getting full gradient for parameters
                        # normal backward will provide only averaged gradient per batch
                        # requires to store output of the layer
                        if len(grad_output[0].shape) == 4:
                            self.weight.full_grad = (grad_output[0] * self.output).sum(-1).sum(-1)
                        else:
                            self.weight.full_grad = (grad_output[0] * self.output)

                    if pruning_method == 6:
                        # implement ICLR2017 paper
                        def just_hook(self, grad_input, grad_output):
                            if len(grad_output[0].shape) == 4:
                                self.weight.full_grad_iclr2017 = (grad_output[0] * self.output).abs().mean(-1).mean(
                                    -1).mean(0)
                            else:
                                self.weight.full_grad_iclr2017 = (grad_output[0] * self.output).abs().mean(0)

                    def forward_hook(self, input, output):
                        self.output = output

                    if not len(pruning_mask_from) > 0:
                        # in case mask is precomputed we remove hooks
                        m_to_add.register_forward_hook(forward_hook)
                        m_to_add.register_backward_hook(just_hook)

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                if pruning_method in [30, 31]:
                    # for densenets.
                    # add previous layer's value for batch norm pruning

                    if isinstance(prev_module, nn.BatchNorm2d):
                        m_to_add = prev_module
                        print(m_to_add, "yes")
                    else:
                        print(m_to_add, "no")

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                if pruning_method in [24, ]:
                    # add previous layer's value for batch norm pruning

                    if isinstance(prev_module, nn.Conv2d):
                        m_to_add = prev_module

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                   "compute_criteria_from": m_to_add.weight}

                if pruning_method in [0, 2, 3, 63]:
                    # add previous layer's value for batch norm pruning
                    if isinstance(prev_module2, nn.Conv2d):
                        print(module_indx, prev_module2, "yes")
                        m_to_add = prev_module2
                    elif isinstance(prev_module2, nn.Linear):
                        print(module_indx, prev_module2, "yes")
                        m_to_add = prev_module2
                    elif isinstance(prev_module, nn.Conv2d):
                        print(module_indx, prev_module, "yes")
                        m_to_add = prev_module
                    elif isinstance(prev_module, nn.Linear):
                        print(module_indx, prev_module, "yes")
                        m_to_add = prev_module
                    elif 'Bottleneck' in str(type(prev_module)):
                        if prev_module.downsample is not None:
                            m_to_add = prev_module.downsample[0]
                            print(module_indx, m_to_add, "yes")
                    else:
                        print(module_indx, m, "no")

                    for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                            "compute_criteria_from": m_to_add.weight}

                pruning_parameters_list.append(for_pruning)

            prev_module2 = prev_module
            prev_module = m

    if model_name == "resnet20":
        # prune only even layers as in Rethinking min norm pruning
        pruning_parameters_list = [d for di, d in enumerate(pruning_parameters_list) if (di % 2 == 1 and di > 0)]

    if ("prune_only_skip_connections" in name) and 1:
        # will prune only skip connections (gates around them). Works with ResNets only
        pruning_parameters_list = pruning_parameters_list[:4]

    return pruning_parameters_list

class ExpMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, mom = 0.9):
        self.reset()
        self.mom = mom

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.exp_avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean_avg = self.sum / self.count
        self.exp_avg = self.mom*self.exp_avg + (1.0 - self.mom)*self.val
        if self.count == 1:
            self.exp_avg = self.val

if __name__ == '__main__':
    pass

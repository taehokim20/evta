# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from multiprocessing import Process
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import json
import torch
from schema import And, Optional

from nni.utils import OptimizeMode

from nni.compression.pytorch.compressor import Pruner
from nni.compression.pytorch.utils.config_validation import CompressorSchema
from nni.compression.pytorch.utils.num_param_counter import get_total_num_weights
from nni.algorithms.compression.pytorch.pruning.constants_pruner import PRUNER_DICT

################### TVM build part addition ###############
from models.cifar10.resnet import ResNet18, ResNet50
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import _pickle as cPickle
import time
import torch.onnx
import onnxruntime
import tensorflow as tf

import socket
import sys 

import tvm
from tvm import relay, auto_scheduler
import numpy as np
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import rpc
from tvm.contrib import utils, ndk, graph_runtime as runtime
from tvm.contrib import graph_executor
from nni.compression.pytorch.utils.counter import count_flops_params

from nni.compression.pytorch import ModelSpeedup
from torch.optim.lr_scheduler import MultiStepLR
import gc
###########################################################


_logger = logging.getLogger(__name__)


class FinalAccuracyMeasurement(Pruner):
    '''
    '''
    def __init__(self, model, config_list, short_term_fine_tuner, evaluator, val_loader, dummy_input, criterion,
                 optimize_mode='maximize', base_algo='l1', sparsity_per_iteration=0.01, experiment_data_dir='./', cpu_or_gpu=1):
        # models used for iterative pruning and evaluation
        self._model_to_prune = copy.deepcopy(model)
        self._original_model = copy.deepcopy(model)
        self._base_algo = base_algo
        self._cpu_or_gpu = cpu_or_gpu

        super().__init__(model, config_list)

        self._short_term_fine_tuner = short_term_fine_tuner
        self._evaluator = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)

        # config_list
        self._config_list_generated = []

        self._experiment_data_dir = experiment_data_dir
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)

        self._tmp_model_path = './tmp_model.pth'

        # addition
        self._val_loader = val_loader
        self._criterion = criterion
        self._dummy_input = dummy_input

    def _update_config_list(self, config_list, op_name, sparsity):
        '''
        update sparsity of op_name in config_list
        '''
        config_list_updated = copy.deepcopy(config_list)
        if not op_name:
            return config_list_updated

        for idx, item in enumerate(config_list):
            if op_name in item['op_names']:
                config_list_updated[idx]['sparsity'] = sparsity
                return config_list_updated

        # if op_name is not in self._config_list_generated, create a new json item
        if self._base_algo in ['l1', 'l2', 'fpgm']:
            config_list_updated.append(
                {'sparsity': sparsity, 'op_types': ['Conv2d'], 'op_names': [op_name]})
        elif self._base_algo == 'level':
            config_list_updated.append(
                {'sparsity': sparsity, 'op_names': [op_name]})

        return config_list_updated

    def compress(self):
        """
        Compress the model.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        
        real_pruning_times = [16, 25, 28, 25, 28, -1, 0, -1, 12, 0, 19, 18, -1, 14, 18, 14, 5, -1, -1, 5]
        layer_idx = 0
        for wrapper in self.get_modules_wrapper():
            if real_pruning_times[layer_idx] > -1:
                target_op_sparsity = (1 + real_pruning_times[layer_idx]) * (1/32)
                self._config_list_generated = self._update_config_list(
                    self._config_list_generated, wrapper.name, target_op_sparsity)
                pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(self._model_to_prune), self._config_list_generated, dependency_aware=True, dummy_input=self._dummy_input)
                model_masked = pruner.compress()
                masks = {}
                for w in pruner.get_modules_wrapper():
                    if w.name == wrapper.name:
                        masks = {'weight_mask': w.weight_mask,
                                 'bias_mask': w.bias_mask}
                        break
                for k in masks:
                    setattr(wrapper, k, masks[k])
            layer_idx += 1
            ######################################################################
        # added 0: speed_up
        pruner.export_model('./model_masked.pth', './mask.pth')
        model = copy.deepcopy(self._original_model)
        model.load_state_dict(torch.load('./model_masked.pth'))
        masks_file = './mask.pth'
        m_speedup = ModelSpeedup(model, self._dummy_input, masks_file, device)
        m_speedup.speedup_model()

        # update weights parameters
        self._model_to_prune.load_state_dict(torch.load(self._tmp_model_path))

        # load weights parameters
        self.load_model_state_dict(torch.load(self._tmp_model_path))

        return self.bound_model

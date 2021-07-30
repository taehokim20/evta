# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
### Error Handling ###
import os
import warnings
warnings.filterwarnings(action='ignore')
import logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
logging.basicConfig(level=logging.WARNING)
absl.logging._warn_preinit_stderr = False
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
######################
'''
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
from .constants_pruner import PRUNER_DICT

################### TVM build part addition ###############
from pruned_vgg_maxpool import VGG 
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
from tvm.contrib import graph_executor                              ## for normal running
#from tvm.contrib.debugger import debug_executor as graph_executor    ## for debugging
#from torchsummary import summary
from nni.compression.pytorch.utils.counter import count_flops_params

from nni.compression.pytorch import ModelSpeedup
import gc
###########################################################


_logger = logging.getLogger(__name__)


class NetAdaptPruner(Pruner):
    """
    A Pytorch implementation of NetAdapt compression algorithm.

    Parameters
    ----------
    model : pytorch model
        The model to be pruned.
    config_list : list
        Supported keys:
            - sparsity : The target overall sparsity.
            - op_types : The operation type to prune.
    short_term_fine_tuner : function
        function to short-term fine tune the masked model.
        This function should include `model` as the only parameter,
        and fine tune the model for a short term after each pruning iteration.
        Example::

            def short_term_fine_tuner(model, epoch=3):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_loader = ...
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                model.train()
                for _ in range(epoch):
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
    evaluator : function
        function to evaluate the masked model.
        This function should include `model` as the only parameter, and returns a scalar value.
        Example::

            def evaluator(model):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                val_loader = ...
                model.eval()
                correct = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        # get the index of the max log-probability
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / len(val_loader.dataset)
                return accuracy
    optimize_mode : str
        optimize mode, `maximize` or `minimize`, by default `maximize`.
    base_algo : str
        Base pruning algorithm. `level`, `l1`, `l2` or `fpgm`, by default `l1`. Given the sparsity distribution among the ops,
        the assigned `base_algo` is used to decide which filters/channels/weights to prune.
    sparsity_per_iteration : float
        sparsity to prune in each iteration.
    experiment_data_dir : str
        PATH to save experiment data,
        including the config_list generated for the base pruning algorithm and the performance of the pruned model.
    """

    def __init__(self, model, config_list, short_term_fine_tuner, evaluator, val_loader, num, dummy_input, criterion,
                 optimize_mode='maximize', base_algo='l1', sparsity_per_iteration=0.01, experiment_data_dir='./'):
        # models used for iterative pruning and evaluation
        self._model_to_prune = copy.deepcopy(model)
        self._base_algo = base_algo

        super().__init__(model, config_list)

        self._short_term_fine_tuner = short_term_fine_tuner
        self._evaluator = evaluator
        self._optimize_mode = OptimizeMode(optimize_mode)

        # hyper parameters for NetAdapt algorithm
        self._sparsity_per_iteration = sparsity_per_iteration

        # overall pruning rate
        self._sparsity = config_list[0]['sparsity']

        # config_list
        self._config_list_generated = []

        self._experiment_data_dir = experiment_data_dir
        if not os.path.exists(self._experiment_data_dir):
            os.makedirs(self._experiment_data_dir)

        self._tmp_model_path = os.path.join(self._experiment_data_dir, 'tmp_model.pth')

        # addition
        self._num = num
        self._val_loader = val_loader
        self._criterion = criterion
        self._dummy_input = dummy_input

    def validate_config(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model to be pruned
        config_list : list
            List on pruning configs
        """

        if self._base_algo == 'level':
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                Optional('op_types'): [str],
                Optional('op_names'): [str],
            }], model, _logger)
        elif self._base_algo in ['l1', 'l2', 'fpgm']:
            schema = CompressorSchema([{
                'sparsity': And(float, lambda n: 0 < n < 1),
                'op_types': ['Conv2d'],
                Optional('op_names'): [str]
            }], model, _logger)

        schema.validate(config_list)

    def calc_mask(self, wrapper, **kwargs):
        return None

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

    def _get_op_num_weights_remained(self, op_name, module):
        '''
        Get the number of weights remained after channel pruning with current sparsity

        Returns
        -------
        int
            remained number of weights of the op
        '''

        # if op is wrapped by the pruner
        for wrapper in self.get_modules_wrapper():
            if wrapper.name == op_name:
                return wrapper.weight_mask.sum().item()

        # if op is not wrapped by the pruner
        return module.weight.data.numel()

    def _get_op_sparsity(self, op_name):
        for config in self._config_list_generated:
            if 'op_names' in config and op_name in config['op_names']:
                return config['sparsity']
        return 0

    def _calc_num_related_weights(self, op_name):
        '''
        Calculate total number weights of the op and the next op, applicable only for models without dependencies among ops

        Parameters
        ----------
        op_name : str

        Returns
        -------
        int
            total number of all the realted (current and the next) op weights
        '''
        num_weights = 0
        flag_found = False
        previous_name = None
        previous_module = None

        for name, module in self._model_to_prune.named_modules():
            if not flag_found and name != op_name and type(module).__name__ in ['Conv2d', 'Linear']:
                previous_name = name
                previous_module = module
            if not flag_found and name == op_name:
                _logger.debug("original module found: %s", name)
                num_weights = module.weight.data.numel()

                # consider related pruning in this op caused by previous op's pruning
                if previous_module:
                    sparsity_previous_op = self._get_op_sparsity(previous_name)
                    if sparsity_previous_op:
                        _logger.debug(
                            "decrease op's weights by %s due to previous op %s's pruning...", sparsity_previous_op, previous_name)
                        num_weights *= (1-sparsity_previous_op)

                flag_found = True
                continue
            if flag_found and type(module).__name__ in ['Conv2d', 'Linear']:
                _logger.debug("related module found: %s", name)
                # channel/filter pruning crossing is considered here, so only the num_weights after channel pruning is valuable
                num_weights += self._get_op_num_weights_remained(name, module)
                break

        _logger.debug("num related weights of op %s : %d", op_name, num_weights)

        return num_weights

    def _test3(self, model, input_name, ctx, text):
        test_loss = 0 
        correct = 0 
        total_time = 0 
        cases = 20
        loc = 0 
        warm_up = 10
        with torch.no_grad():
            for data, target in self._val_loader:
                loc = loc + 1 
                if loc % 5 == 0:
                    print(loc)
                if loc == cases + 1:
                    break
                output_arr = np.array([1,2])
                for i in range(len(target)):
                    model.set_input(input_name, np.expand_dims(data[i], 0)) 
                    t0 = time.time()
                    model.run()
                    t1 = time.time()
                    if loc > warm_up:
                        total_time += (t1 - t0)  
                    output = model.get_output(0)
                    output = output.asnumpy()
                    output = np.ravel(output, order='C')
                    if i == 0:
                        output_arr = output
                    else:
                        output_arr = np.append(output_arr, output, axis=0)
                output_arr = output_arr.reshape(len(target), 10) 
                output_arr = torch.from_numpy(output_arr)
                # sum up batch loss
                test_loss += self._criterion(output_arr, target).item()
                # get the index of the max log-probability
                pred = output_arr.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        real_cases = (cases - warm_up) * len(target)
#        test_loss /= real_cases
#        accuracy = correct / real_cases
        latency = (total_time*1000) / real_cases
#        fps2 = real_cases / total_time
        print('{} Latency: {:.6f}'.format(text, latency))
#        print('{} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Latency: {:.6f}'.format(
#                    text, test_loss, correct, real_cases, 100. * accuracy, latency))

        return latency

    def compress(self):
        """
        Compress the model.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        _logger.info('Starting NetAdapt Compression...')
        from PIL import Image
        from tvm.contrib.download import download_testdata
        img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
        img_path = download_testdata(img_url, "cat.png", module="data")
        img = Image.open(img_path).resize((32, 32))
        from torchvision import transforms
        my_preprocess = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),   
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )
        img = my_preprocess(img)
        img = np.expand_dims(img, 0)

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        arch = "arm64"
        target = "llvm -mtriple=%s-linux-android" % arch        
#        target = "opencl --device=mali"
#        target_host = "llvm -mtriple=arm64-linux-android"
################# Autotune added
        network = "vgg"
        device_key = "android"
        log_file = "%s.%s.log" % (device_key, network)
        dtype = "float32"
        use_android = True
################################
        self._model_to_prune.eval()
        _, _, temp_results = count_flops_params(self._model_to_prune, (1, 3, 32, 32))
        input_shape = [1, 3, 32, 32]
        output_shape = [1, 10]
        input_data = torch.randn(input_shape).to(device)
        scripted_model = torch.jit.trace(self._model_to_prune, input_data).eval()
        input_name = "input0"
        shape_list = [(input_name, img.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        ########### NCHW -> NHWC ############
        desired_layouts = {'nn.conv2d': ['NHWC', 'default'], 'nn.dense': ['NHWC', 'default']}
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts),
                                        relay.transform.InferType(),
                                        relay.transform.FoldConstant(),
#                                        relay.transform.DebugPrint(),
                                        relay.transform.DeadCodeElimination()])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        #####################################
        tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
        tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9191))

        #################### Extract search tasks ###################
        print("Extract tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
#        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target="opencl --device=mali", target_host=target)
        print("=============== Weights check ==============")
        print(task_weights)

        for idx, task in enumerate(tasks):            
            print("============ Task %d (workload key: %s) ===========" % (idx, task.workload_key))
            print(task.compute_dag)        
        
        #################### Tuning #####################
        print("Begin tuning...")
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)

        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,
            builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_android else "default"),
            runner=auto_scheduler.RPCRunner(device_key, host=tracker_host, port=tracker_port, timeout=10000, repeat=1, min_repeat_ms=200, enable_cpu_cache_flush=True,),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
	    verbose=1,
            early_stopping=24,
        )
        tuner.tune(tune_option)#, per_task_early_stopping=30)
        
        #################### Compile ####################
        print("Compile...")
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build_module.build(mod, params=params, target=target)
#                lib = relay.build(mod, params=params, target="opencl", target_host=target)
            
        tmp = utils.tempdir()
        lib_fname = tmp.relpath("net.so")
        lib.export_library(lib_fname, ndk.create_shared)
        remote = auto_scheduler.utils.request_remote(device_key, tracker_host, tracker_port, timeout=10000)
        remote.upload(lib_fname)
        rlib = remote.load_module("net.so")

        # Create graph executor
        ctx = remote.cpu()
#        ctx = remote.cl(0)
        module = graph_executor.GraphModule(rlib["default"](ctx))
#        module = graph_executor.GraphModuleDebug(rlib["default"](ctx), [ctx], rlib.get_json(), dump_root="./tvmdbg")

        current_latency = self._test3(module, input_name, ctx, "TVM_initial")
#        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
#        module.set_input(input_name, data_tvm)
#        print("Evaluate inference time cost...")
#        ftimer = module.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=500)
#        prof_res = np.array(ftimer().results) * 1e3
#        print("Mean inference time (std dev): %8.4f ms (%8.4f ms)" %(np.mean(prof_res), np.std(prof_res)))
#        current_latency = np.mean(prof_res)
        #################################################
        
        pruning_iteration = 0
        delta_num_weights_per_iteration = \
            int(get_total_num_weights(self._model_to_prune, ['Conv2d', 'Linear']) * self._sparsity_per_iteration)
        init_resource_reduction_ratio = 0.025 # 0.05 
        resource_reduction_decay = 0.96 #0.98
        max_iter = 30

        budget = 0.5 * current_latency
        init_resource_reduction = init_resource_reduction_ratio * current_latency
        print('Current latency: {:>8.4f}'.format(current_latency))
        file_object = open('./record_tvm.txt', 'a')
        file_object.write('Budget: {:>8.4f}, Current latency: {:>8.4f}\n'.format(budget, current_latency))
        file_object.close()
        current_accuracy = self._evaluator(self._model_to_prune)
        pass_target_latency = 0
        target_latency = current_latency - init_resource_reduction * (resource_reduction_decay ** (pruning_iteration - 1))

        # stop condition
        while pruning_iteration < max_iter and current_latency > budget:
            _logger.info('Pruning iteration: %d', pruning_iteration)


            # calculate target sparsity of this iteration
#            target_sparsity = current_sparsity + self._sparsity_per_iteration
            if pass_target_latency == 1:
                target_latency = current_latency - init_resource_reduction * (resource_reduction_decay ** (pruning_iteration - 1))
                pass_target_latency = 0

            # Print the message
            print('=======================')
            print(('Process iteration {:>3}: current_accuracy = {:>8.3f}, '
                    'current_latency = {:>8.3f}, target_latency = {:>8.3f} \n').format(pruning_iteration, current_accuracy, current_latency, target_latency))            
            file_object = open('./record_tvm.txt', 'a')            
            file_object.write(('Process iteration {:>3}: current_accuracy = {:>8.3f}, '
                   'current_latency = {:>8.3f}, target_resource = {:>8.3f} \n').format(pruning_iteration, current_accuracy, current_latency, target_latency))
            file_object.close()

            # variable to store the info of the best layer found in this iteration
            best_op = {}
            layer_idx = 1
            total_channel = 0
            times_num = 4

            for wrapper in self.get_modules_wrapper():
#                if pruning_iteration == 0:
#                    self._config_list_generated = self._update_config_list(
#                        self._config_list_generated, wrapper.name, 0.375)
#                # update mask of the chosen op
#                config_list = self._update_config_list(self._config_list_generated, wrapper.name, target_op_sparsity)
#                pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(self._model_to_prune), config_list)
#                model_masked = pruner.compress()
#                for w in pruner.get_modules_wrapper():
#                    if w.name == wrapper.name:
#                        masks = {'weight_mask': w.weight_mask,
#                                 'bias_mask': w.bias_mask}
#                        break
#                for k in masks:
#                    setattr(wrapper, k, masks[k])
#                continue

#                for wrapper in self.get_modules_wrapper():
#                    if wrapper.name == best_op['op_name']:
#                        for k in best_op['masks']:
#                            setattr(wrapper, k, best_op['masks'][k])
#                        break
#                    continue
                current_op_sparsity = 1 - wrapper.weight_mask.sum().item() / wrapper.weight_mask.numel()
                # sparsity that this layer needs to prune to satisfy the requirement
#                target_op_sparsity = current_op_sparsity + delta_num_weights_per_iteration / self._calc_num_related_weights(wrapper.name)

                if layer_idx > 7:
                    total_channel = 512 
                    times_num = 16
                elif layer_idx > 4:
                    total_channel = 256 
                    times_num = 8
                elif layer_idx > 2:
                    total_channel = 128 
                    times_num = 4
                else:
                    total_channel = 64
                    times_num = 2
                target_op_sparsity = 0.375 + pruning_iteration * (times_num / total_channel)
                layer_idx += 1
                if layer_idx <= 3 and pruning_iteration % 2 == 0:
                    file_object = open('./record_tvm.txt', 'a')
                    file_object.write('NOT pruning 2 channels: ' + wrapper.name + '\n')
                    file_object.close()
                    continue

                if target_op_sparsity >= 0.95:
                    print('Improper Layer')
                    file_object = open('./record_tvm.txt', 'a')
                    file_object.write('Improper Layer: ' + wrapper.name + '\n')
                    file_object.close()
                    continue

                config_list = self._update_config_list(self._config_list_generated, wrapper.name, target_op_sparsity)
                pruner = PRUNER_DICT[self._base_algo](copy.deepcopy(self._model_to_prune), config_list)
                model_masked = pruner.compress()
                ch_num = int(total_channel * (1 - target_op_sparsity))

                # added 0: speed_up
                pruner.export_model('./model_masked.pth', './mask.pth')
                model = VGG(depth=self._num).to(device)
                model.load_state_dict(torch.load('./model_masked.pth'))
                masks_file = './mask.pth'
                m_speedup = ModelSpeedup(model, self._dummy_input, masks_file, device)
                m_speedup.speedup_model()
                # added 1: Autotune + TVM build
                model.eval()
                _, _, _ = count_flops_params(model, (1, 3, 32, 32))
                input_shape = [1, 3, 32, 32]
                output_shape = [1, 10]
                input_data = torch.randn(input_shape).to(device)
#                scripted_model = torch.jit.trace(self._model_to_prune, input_data).eval()
                scripted_model = torch.jit.trace(model, input_data).eval()
                input_name = "input0"
                shape_list = [(input_name, img.shape)]
                mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
                ########### NCHW -> NHWC ############
                desired_layouts = {'nn.conv2d': ['NHWC', 'default'], 'nn.dense': ['NHWC', 'default']} # added
                seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                      relay.transform.ConvertLayout(desired_layouts)])
                with tvm.transform.PassContext(opt_level=3):
                    mod = seq(mod)
                #####################################
                tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
                tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
                #################### Extract search tasks ###################
                print("Extract tasks...")
                tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
                print("=============== Weights check ==============")
                print(task_weights)

                for idx, task in enumerate(tasks):
                    print("============ Task %d (workload key: %s) ===========" % (idx, task.workload_key))
                    print(task.compute_dag)
                #################### Tuning #####################
                print("Begin tuning...")
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                tune_option = auto_scheduler.TuningOptions(
                    num_measure_trials=200, #5000,
                    builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_android else "default"),
                    runner=auto_scheduler.RPCRunner(device_key, host=tracker_host, port=tracker_port, timeout=10000, repeat=1, min_repeat_ms=200, enable_cpu_cache_flush=True,),
                    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                )
                tuner.tune(tune_option, per_task_early_stopping=30)
                #################### Compile ####################
                print("Compile...")
                with auto_scheduler.ApplyHistoryBest(log_file):
                    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                        lib = relay.build(mod, target=target, params=params)
          
                tmp = utils.tempdir()
                lib_fname = tmp.relpath("net.so")
                lib.export_library(lib_fname, ndk.create_shared)
                remote = auto_scheduler.utils.request_remote(device_key, tracker_host, tracker_port, timeout=10000)
                remote.upload(lib_fname)
                rlib = remote.load_module("net.so")
                ctx = remote.cpu()
#                ctx = remote.cl()
                module = graph_executor.GraphModule(rlib["default"](ctx))

                temp_latency = self._test3(module, input_name, ctx, "TVM_initial")
#                data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
#                module.set_input(input_name, data_tvm)
#                print("Evaluate inference time cost...")
#                ftimer = module.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=500)
#                prof_res = np.array(ftimer().results) * 1e3
#                print("Mean inference time (std dev): %8.4f ms (%8.4f ms)" %(np.mean(prof_res), np.std(prof_res)))
#                temp_latency = np.mean(prof_res)
                #################################################
                print('Layer: {}, Temp latency: {:>8.4f}, Channel: {:4d}'.format(wrapper.name, temp_latency, ch_num))
                file_object = open('./record_tvm.txt', 'a')
                file_object.write('Layer: {}, Temp latency: {:>8.4f}, Channel: {:4d}\n'.format(wrapper.name, temp_latency, ch_num))
                file_object.close()

                if temp_latency <= target_latency:
                    # Short-term fine tune the pruned model
                    self._short_term_fine_tuner(model_masked, epochs=10) #2
                    performance = self._evaluator(model_masked)
                    print('Layer: {}, Evaluation result after short-term fine tuning: {:>8.4f}'.format(wrapper.name, performance))
                    file_object = open('./record_tvm.txt', 'a')
                    file_object.write('Layer: {}, Accuracy: {:>8.4f}\n'.format(wrapper.name, performance))
                    file_object.close()

                if temp_latency <= target_latency and \
                    ( not best_op \
                    or (self._optimize_mode is OptimizeMode.Maximize and performance >= best_op['performance']) \
                    or (self._optimize_mode is OptimizeMode.Minimize and performance < best_op['performance'])):
                    _logger.debug("updating best layer to %s...", wrapper.name)
                    pass_target_latency = 1
                    # find weight mask of this layer
                    for w in pruner.get_modules_wrapper():
                        if w.name == wrapper.name:
                            masks = {'weight_mask': w.weight_mask,
                                     'bias_mask': w.bias_mask}
                            break
                    best_op = {
                        'op_name': wrapper.name,
                        'sparsity': target_op_sparsity,
                        'ch_num': ch_num,
                        'latency': temp_latency,
                        'performance': performance,
                        'masks': masks
                    }

                    current_latency = temp_latency

                    # save model weights
                    pruner.export_model(self._tmp_model_path)

#                print('Relaunch app!')
#                time.sleep(60)

#            if not best_op:
#                # decrease pruning step
#                self._sparsity_per_iteration *= 0.5
#                _logger.info("No more layers to prune, decrease pruning step to %s", self._sparsity_per_iteration)
#                pruning_iteration = max_iter
#                continue

            if pass_target_latency == 1:
                # Pick the best layer to prune, update iterative information
                # update config_list
                self._config_list_generated = self._update_config_list(
                    self._config_list_generated, best_op['op_name'], best_op['sparsity'])

                # update weights parameters
                self._model_to_prune.load_state_dict(torch.load(self._tmp_model_path))
                print('Budget: {:>8.4f}, Current latency: {:>8.4f}'.format(budget, best_op['latency']))
                file_object = open('./record_tvm.txt', 'a')
                file_object.write('Budget: {:>8.4f}, Current latency: {:>8.4f} \n'.format(budget, best_op['latency']))

                # update mask of the chosen op
                for wrapper in self.get_modules_wrapper():
                    if wrapper.name == best_op['op_name']:
                        for k in best_op['masks']:
                            setattr(wrapper, k, best_op['masks'][k])
                        break

                file_object.write('Layer {} selected with {:4d} channels, latency {:>8.4f}, accuracy {:>8.4f} \n'.format(best_op['op_name'], best_op['ch_num'], best_op['latency'], best_op['performance']))
                file_object.close()
#                current_sparsity = target_sparsity
#                _logger.info('Pruning iteration %d finished, current sparsity: %s', pruning_iteration, current_sparsity)
                _logger.info('Layer %s seleted with sparsity %s, performance after pruning & short term fine-tuning : %s', best_op['op_name'], best_op['sparsity'], best_op['performance'])
                self._final_performance = best_op['performance']
            pruning_iteration += 1

        # load weights parameters
        self.load_model_state_dict(torch.load(self._tmp_model_path))
#        os.remove(self._tmp_model_path)

        _logger.info('----------Compression finished--------------')
        _logger.info('config_list generated: %s', self._config_list_generated)
        _logger.info("Performance after pruning: %s", self._final_performance)
#        _logger.info("Masked sparsity: %.6f", current_sparsity)

        # save best config found and best performance
        with open(os.path.join(self._experiment_data_dir, 'search_result.json'), 'w') as jsonfile:
            json.dump({
                'performance': self._final_performance,
                'config_list': json.dumps(self._config_list_generated)
            }, jsonfile)

        _logger.info('search history and result saved to foler : %s', self._experiment_data_dir)

        return self.bound_model

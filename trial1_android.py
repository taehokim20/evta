import torch
import time
from torchvision import datasets, transforms
from models.vgg_maxpool import VGG
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import _pickle as cPickle
import tensorflow as tf
import tvm
from tvm import relay
import numpy as np
from tvm import autotvm
import tvm.relay.testing
from tvm autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as graph_runtime
from tvm import rpc
from tvm.contrib import utils, ndk

def get_data(dataset, data_dir, batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    if dataset == 'cifar10':
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()


    return train_loader, val_loader, criterion

def test_tflite(interpreter, criterion, val_loader):
    test_loss = 0
    correct = 0
    total_time = 0
    for data, target in val_loader:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], data)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        total_time += (t1-t0)
        output = interpreter.get_tensor(output_details[0]['index'])
        #output = torch.tensor(output[0], dtype=torch.float32)
        output = torch.from_numpy(output)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    fps = len(val_loader.dataset) / total_time
    print('TF TensorFlow Lite Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), FPS: {:.2f}'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy, fps))

    return accuracy, fps

def test3(model, input_name, ctx, criterion, val_loader, text):
    ftimer = model.module.time_evaluator("run", ctx, number=1, repeat=600)
    fps = 1 / np.mean(np.array(ftimer().results))

    test_loss = 0 
    correct = 0 
    total_time = 0 
    with torch.no_grad():
        for data, target in val_loader:
            output_arr = np.array([1,2])
            for i in range(len(target)):
                model.set_input(input_name, np.expand_dims(data[i], 0)) 
                t0 = time.time()
                model.run()
                t1 = time.time()
                total_time += (t1-t0)
                output = model.get_output(0)
                output = output.asnumpy()
                output = np.ravel(output, order='C')
                if i==0:
                    output_arr = output
                else:
                    output_arr = np.append(output_arr, output, axis=0)
            output_arr = output_arr.reshape(len(target), 10) 
            output_arr = torch.from_numpy(output_arr)
            # sum up batch loss
            test_loss += criterion(output_arr, target).item()
            # get the index of the max log-probability
            pred = output_arr.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    fps2 = len(val_loader.dataset) / total_time
    print('{} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), FPS_1: {:.2f}, FPS_2: {:.2f}'.format(
    	text, test_loss, correct, len(val_loader.dataset), 100. * accuracy, fps, fps2))

    return accuracy

data_dir = './data/'
dataset = 'cifar10'
test_batch_size = 64
tflite_file = '/github/evta/model.tflite'

train_loader, val_loader, criterion = get_data(dataset, data_dir, test_batch_size, test_batch_size)
device = torch.device("cpu")

interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()
test_tflite(interpreter, criterion, val_loader)

#####################################################
# Compile the model with relay
# ----------------------------

local_demo = False #True
test_target = "cpu"
arch = "arm64"
target = "llvm -mtriple=%s-linux-android" % arch
target_host = None
num_outputs = 1

if local_demo:
    target_host = None
    target = "llvm"
elif test_target == "opencl":
    target_host = target
    target = "opencl"
elif test_target == "vulkan":
    target_host = target
    target = "vulkan"

# input_name = "input_1"
# shape_dict = {input_name: x.shape}

my_shape = cPickle.load(open('/github/evta/my_shape.p','rb'))
torch_model = VGG(my_shape=my_shape, depth=16).to(device)
torch_model.load_state_dict(torch.load('/github/evta/model_trained.pth'))
torch_model.eval()

import tvm.contrib.graph_runtime as runtime
input_shape = [1, 3, 32, 32]
output_shape = [1, 10]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(torch_model, input_data).eval()

input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

with tvm.transform.PassContext(opt_level=3):
	lib = relay.build_module.build(mod, target=target, params=params)
path_lib = "tvm_lib_android.tar"
fcompile = ndk.create_shared if not local_demo else None
lib.export_library(path_lib, fcompile)
#loaded_lib = tvm.runtime.load_module(path_lib)
#ctx = tvm.context(str(target), 0)
#module = runtime.GraphModule(loaded_lib["default"](ctx))

# test3(module, input_name, ctx, criterion, val_loader, "TVM")

tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
key = "android"

if local_demo:
	remote = rpc.LocalSession()
else:
	tracker = rpc.connect_tracker(tracker_host, tracker_port)
	remote = tracker.request(key, priority=0, session_timeout=60)

if local_demo:
	ctx = remote.cpu(0)
elif test_target == "opencl":
	ctx = remote.cl(0)
elif test_target == "vulkan":
	ctx = remote.vulkan(0)
else:
	ctx = remote.cpu(0)

remote.upload("tvm_lib_android.so")
rlib = remote.load_module("tvm_lib_android.so")
module = runtime.GraphModule(rlib["default"](ctx))

test3(module, input_name, ctx, criterion, val_loader, "TVM") #####

# for i in range(num_outputs):
# 	print('==========TEST MODEL #{ } ==========='.format(str(i)))
# 	my_shape = cPickle.load(open('/github/evta/my_shape.p','rb'))
# 	torch_model = VGG(my_shape=my_shape, depth=16).to(device)
# 	torch_model.load_state_dict(torch.load('/github/evta/model_trained.pth'))
# 	torch_model.eval()

# 	#### tvm ####
# 	import tvm.contrib.graph_runtime as runtime
# 	input_shape = [1, 3, 32, 32]

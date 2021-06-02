
### Error Handling ###

import os
import warnings
import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)
'''
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
'''
import torch
import time
from torchvision import datasets, transforms
from models.vgg_maxpool import VGG
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import _pickle as cPickle
import tensorflow as tf
import tvm
from tvm import relay, autotvm
import numpy as np
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import rpc
from tvm.contrib import utils, ndk, graph_runtime as runtime

num = 11

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
    loc = 0
    for data, target in val_loader:
        loc = loc + 1
        if loc==641:
            break
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

#    test_loss /= len(val_loader.dataset)
#    accuracy = correct / len(val_loader.dataset)
#    fps = len(val_loader.dataset) / total_time
    test_loss /= 640
    accuracy = correct / 640
    fps = 640 / total_time
    print('TF TensorFlow Lite Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), FPS: {:.2f}'.format(
        test_loss, correct, 640, 100. * accuracy, fps))

    return accuracy, fps

def test3(model, input_name, ctx, criterion, val_loader, text):
#    ftimer = model.module.time_evaluator("run", ctx, number=1, repeat=600)
#    fps = 1 / np.mean(np.array(ftimer().results))

    test_loss = 0 
    correct = 0 
    total_time = 0 
    loc = 0
    with torch.no_grad():
        for data, target in val_loader:
            loc = loc + 1
#            if loc % 10 == 0:
#                print(loc)
            if loc==641:
                break
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
#            print('{}'.format(total_time))
            output_arr = output_arr.reshape(len(target), 10) 
            output_arr = torch.from_numpy(output_arr)
            # sum up batch loss
            test_loss += criterion(output_arr, target).item()
            # get the index of the max log-probability
            pred = output_arr.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

#    test_loss /= len(val_loader.dataset)
#    accuracy = correct / len(val_loader.dataset)
#    fps2 = len(val_loader.dataset) / total_time
    test_loss /= 640
    accuracy = correct / 640
    fps2 = 640 / total_time
    print('{} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), FPS: {:.2f}'.format(
    	text, test_loss, correct, 640, 100. * accuracy, fps2))

    return accuracy

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


data_dir = './data/'
dataset = 'cifar10'
test_batch_size = 1 #64
tflite_file = os.path.join('/github/evta2/output', str(num), 'model.tflite')

train_loader, val_loader, criterion = get_data(dataset, data_dir, test_batch_size, test_batch_size)
device = torch.device("cpu")
#device = torch.device("cuda")

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

interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()
test_tflite(interpreter, criterion, val_loader)

#####################################################
# Compile the model with relay
# ----------------------------

use_android = True
local_demo = False
test_target = "opencl"
#target = "llvm -mtriple=arm64-linux-android"
#target = tvm.target.Target("opencl -device=adreno")
target = 'opencl --device=mali'
#target = tvm.target.Target("opencl --device=mali")
#target_host = "llvm -mtriple=adreno"
target_host = "llvm -mtriple=arm64-linux-android"
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
my_shape = cPickle.load(open(os.path.join('/github/evta2/output', str(num), 'my_shape.p'),'rb'))
torch_model = VGG(my_shape=my_shape, depth=16).to(device)
#torch_model = VGG(my_shape=my_shape, depth=16)
torch_model.load_state_dict(torch.load(os.path.join('/github/evta2/output', str(num), 'model_trained.pth'),map_location=torch.device('cpu')))
torch_model.eval()
input_shape = [1, 3, 32, 32]
output_shape = [1, 10]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(torch_model, input_data).eval()
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list) 
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build_module.build(mod, target=target, params=params, target_host=target_host)
#    lib = relay.build_module.build(mod, target=target, params=params)

# export library
tmp = utils.tempdir()
if use_android:
    from tvm.contrib import ndk
    lib_fname = tmp.relpath("net.so")    
    lib.export_library(lib_fname, ndk.create_shared)
else:
    lib_fname = tmp.relpath("net.tar")
    lib.export_library(lib_fname)
tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
key = "android"

tracker = rpc.connect_tracker(tracker_host, tracker_port)
remote = tracker.request(key, priority=0, session_timeout=200000)

if local_demo:
    ctx = remote.cpu(0)
elif test_target == "opencl":
    ctx = remote.cl(0)
elif test_target == "vulkan":
    ctx = remote.vulkan(0)
else:
    ctx = remote.cpu(0)

remote.upload(lib_fname)
rlib = remote.load_module("net.so")
module = runtime.GraphModule(rlib["default"](ctx))
'''
# upload module to device
print("Upload...")
#remote = autotvm.measure.request_remote(key, "0.0.0.0", 9190, timeout=200000)
remote.upload(tmp.relpath("net.so"))
rlib = remote.load_module("net.so")

# upload parameters to device
ctx = remote.context(str(target), 0)
module = runtime.GraphModule(rlib["default"](ctx))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)
'''
## evaluate
#test3(module, input_name, ctx, criterion, val_loader, "TVM") #####


############### Autotune
network = "vgg-16"
device_key = "android"
log_file = "%s.%s.log" % (device_key,network)
dtype = "float32"

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 100,
    "early_stopping": 50,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
        runner=autotvm.RPCRunner(device_key, host="0.0.0.0", port=9190, number=10, timeout=5,),
    ),
}

tasks = autotvm.task.extract_from_program(
    mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
)
tune_tasks(tasks, **tuning_option)
with autotvm.apply_history_best(log_file):
    with tvm.transform.PassContext(opt_level=3):
        lib=relay.build_module.build(mod, target=target, params=params)
    tmp = utils.tempdir()
    lib_fname = tmp.relpath("net.so")
    fcompile = ndk.create_shared if not local_demo else None
    lib.export_library(lib_fname, fcompile)
    remote.upload(lib_fname)
    rlib = remote.load_module("net.so")
    ctx = remote.context(str(target), 0)
    module = runtime.GraphModule(rlib["default"](ctx))

    test3(module, input_name, ctx, criterion, val_loader, "Autotune")


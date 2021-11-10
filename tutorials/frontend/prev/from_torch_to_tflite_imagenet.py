import torch
import time
from torchvision import datasets, transforms
#from pruned_vgg_maxpool import VGG
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from models.cifar10.resnet import ResNet18, ResNet50
import torchvision.models as models
from torchvision.transforms.functional import InterpolationMode

import random
import torch.onnx
import onnxruntime
import numpy as np
import os
import _pickle as cPickle

import tensorflow as tf
'''
tf.get_logger().setLevel('ERROR')
import warnings
import logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logging.getLogger('tensorflow').disabled = True
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)
'''
filename = 'mobilenetv2_imagenet'

def get_data(dataset, data_dir, batch_size, test_batch_size):
    '''
    get data
    '''
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
    elif dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        interpolation = InterpolationMode.BILINEAR
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                 transform=transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=test_batch_size, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
    return train_loader, val_loader, criterion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tflite_path = os.path.join('./temp', filename + '.tflite')
data_dir = './data'
dataset = 'imagenet' #'cifar10'
batch_size = 16
test_batch_size = 1

_, val_loader, criterion = get_data(dataset, data_dir, batch_size, test_batch_size)
model = models.mobilenet_v2(pretrained=True).to(device)

x, _ = next(iter(val_loader))

model.eval()

torch_out = model(x.to(device))
torch.onnx.export(model, x.to(device), "test.onnx", export_params=True, opset_version=10,
                  do_constant_folding=True, input_names=['input'], output_names=['output'])

from onnx_tf.backend import prepare
import onnx

onnx_model = onnx.load("test.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("./tf_export")

import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()

converter = tf_v1.lite.TFLiteConverter.from_saved_model("./tf_export")
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf_v1.lite.OpsSet.TFLITE_BUILTINS, tf_v1.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()
open(tflite_path, "wb").write(tflite_model)


import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], x)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import time
import json
import torch
import torch.utils.data
from torch import nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision

#from models.cifar10.vgg import VGG
from models.cifar10.resnet import ResNet18, ResNet50
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from inter_tuner import InterTuner
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
import _pickle as cPickle
from torchsummary import summary
from datetime import datetime

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
            batch_size=batch_size, shuffle=False, **kwargs)
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


def train(args, model, device, train_loader, criterion, optimizer, epoch, callback=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            file_object = open('./train_epoch.txt', 'a')
            file_object.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            file_object.close()

def test(model, device, criterion, val_loader):
    model.eval()
    total_len = len(val_loader.dataset)
    test_loss = 0
    correct = 0
    correct_5 = 0
    count = 0
    with torch.no_grad():
        for data, target in val_loader:
            count += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            _, pred = output.topk(5, 1, True, True)
            temp_1 = pred.eq(target.view(1, -1).expand_as(pred))
            temp_5 = temp_1[:5].view(-1)
            correct_5 += temp_5.sum().item()
            if count % 5000 == 0 and count != total_len:
                print('Top-1: {}/{} ({:.4f}%), Top-5: {}/{} ({:.4f}%)'.format(correct, count, 100.*(correct/count), correct_5, count, 100.*(correct_5/count)))

    test_loss /= total_len
    accuracy = correct / total_len
    accuracy_5 = correct_5 / total_len

    print('\nTest set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.4f}%), Top-5 Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, total_len, 100. * accuracy, correct_5, total_len, 100. * accuracy_5))
    file_object = open('./train_epoch.txt', 'a')
    file_object.write('Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.4f}%), Top-5 Accuracy: {}/{} ({:.4f}%)\n\n'.format(
        test_loss, correct, total_len, 100. * accuracy, correct_5, total_len, 100. * accuracy_5))
    file_object.close()

    return accuracy, accuracy_5

def get_dummy_input(args, device):
    if args.dataset == 'mnist':
        dummy_input = torch.randn([args.test_batch_size, 1, 28, 28]).to(device)
    elif args.dataset == 'cifar10':
        dummy_input = torch.randn([args.test_batch_size, 3, 32, 32]).to(device)
    elif args.dataset == 'imagenet':
        dummy_input = torch.randn([args.test_batch_size, 3, 224, 224]).to(device)
    return dummy_input


def get_input_size(dataset):
    if dataset == 'mnist':
        input_size = (1, 1, 28, 28)
    elif dataset == 'cifar10':
        input_size = (1, 3, 32, 32)
    elif dataset == 'imagenet':
        input_size = (1, 3, 224, 224)
    return input_size


def main(args):
    cpu_or_gpu = 1 #1: cpu, 2: gpu
    # prepare dataset
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)
    # model, optimizer = get_trained_model_optimizer(args, device, train_loader, val_loader, criterion)

    # model = ResNet50().to(device) #VGG(depth=16).to(device)
    # model.load_state_dict(torch.load('./model_trained.pth'))
    model = models.resnet18(pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4) # lr=1e-4

    def short_term_fine_tuner(model, optimizer=optimizer, epochs=1):
        train(args, model, device, train_loader, criterion, optimizer, epochs)

    def trainer(model, optimizer, criterion, epoch, callback):
        return train(args, model, device, train_loader, criterion, optimizer, epoch=epoch, callback=callback)

    def evaluator(model):
        return test(model, device, criterion, val_loader)

    # module types to prune, only "Conv2d" supported for channel pruning
    if args.base_algo in ['l1', 'l2', 'fpgm']:
        op_types = ['Conv2d']
    elif args.base_algo == 'level':
        op_types = ['default']

    config_list = [{
        'sparsity': args.sparsity,
        'op_types': op_types
    }]
    dummy_input = get_dummy_input(args, device)
#    pruner = InterTuner(model, config_list, short_term_fine_tuner=short_term_fine_tuner, evaluator=evaluator, val_loader=val_loader, dummy_input=dummy_input, criterion=criterion, base_algo=args.base_algo, experiment_data_dir=args.experiment_data_dir, cpu_or_gpu=cpu_or_gpu)

    # Pruner.compress() returns the masked model
#    original_model = model
#    model = pruner.compress()
#    accuracy, accuracy_5 = evaluator(model)
#    print('Evaluation result (masked model): %s, %s' %(accuracy, accuracy_5))
#    result['performance']['pruned'] = accuracy_5

#    if args.save_model:
#        pruner.export_model(
#            os.path.join(args.experiment_data_dir, 'model_masked.pth'), os.path.join(args.experiment_data_dir, 'mask.pth'))
#        print('Masked model saved to %s' % args.experiment_data_dir)
    
    # model speed up
    if args.speed_up:
        model.load_state_dict(torch.load('./tmp_model.pth'))
        masks_file = './tmp_mask.pth'
        m_speedup = ModelSpeedup(model, dummy_input, masks_file, device)
        m_speedup.speedup_model()
        #model.load_state_dict(torch.load('./model_fine_tuned.pth'))

#        torch.save(model.state_dict(),'model_speed_up.pth')
#        print('Speed up model saved to %s' % args.experiment_data_dir)
    
    if args.fine_tune:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
        #scheduler = MultiStepLR(optimizer, milestones=[int(args.fine_tune_epochs*0.25), int(args.fine_tune_epochs*0.5)], gamma=0.1)
        scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
        best_acc = 0
        best_acc_5 = 0
        for epoch in range(20):
            train(args, model, device, train_loader, criterion, optimizer, epoch)
            scheduler.step()
            acc, acc_5 = evaluator(model)
            if acc_5 > best_acc_5:
                best_acc_5 = acc_5
                torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth'))
            if acc > best_acc:
                best_acc = acc

    print('Evaluation result (fine tuned): %s %s' %(best_acc, best_acc_5))
    print('Fined tuned model saved to %s' % args.experiment_data_dir)
    
#    file_object = open('./record_tvm.txt', 'a')
#    file_object.write('Finish: {}\n'.format(datetime.now()))
#    file_object.close()

#    with open(os.path.join(args.experiment_data_dir, 'result.json'), 'w+') as f:
#        json.dump(result, f)


if __name__ == '__main__':
    def str2bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if s.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='PyTorch Example for SimulatedAnnealingPruner')

    # dataset and model
    parser.add_argument('--dataset', type=str, default= 'imagenet', #'cifar10',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data_fast/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model to use, vgg16, resnet18 or resnet50')
    parser.add_argument('--load-pretrained-model', type=str2bool, default=False,
                        help='whether to load pretrained model')
    parser.add_argument('--pretrained-model-dir', type=str, default='./',
                        help='path to pretrained model')
    parser.add_argument('--pretrain-epochs', type=int, default=2,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, #64
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--fine-tune', type=str2bool, default=True,
                        help='whether to fine-tune the pruned model')
    parser.add_argument('--fine-tune-epochs', type=int, default=3,
                        help='epochs to fine tune')
    parser.add_argument('--experiment-data-dir', type=str, default='./',
                        help='For saving experiment data')

    # pruner
    parser.add_argument('--base-algo', type=str, default='l1',
                        help='base pruning algorithm. level, l1, l2, or fpgm')
    parser.add_argument('--sparsity', type=float, default=0.1,
                        help='target overall target sparsity')
    # param for SimulatedAnnealingPruner
    parser.add_argument('--cool-down-rate', type=float, default=0.9,
                        help='cool down rate')
    # param for NetAdaptPruner
    parser.add_argument('--sparsity-per-iteration', type=float, default=0.05,
                        help='sparsity_per_iteration of NetAdaptPruner')

    # speed-up
    parser.add_argument('--speed-up', type=str2bool, default=True,
                        help='Whether to speed-up the pruned model')

    # others
    parser.add_argument('--log-interval', type=int, default=1000, #200,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str2bool, default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()

    if not os.path.exists(args.experiment_data_dir):
        os.makedirs(args.experiment_data_dir)

    main(args)

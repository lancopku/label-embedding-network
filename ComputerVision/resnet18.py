'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from time import time

from models.resnet18 import *
from utils import progress_bar
from torch.autograd import Variable
import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--tau', default=2, type=float, help='Softmax temperature')
parser.add_argument('--alpha', default=0.9, type=float, help='alpha')
parser.add_argument('--beta', default=0.5, type=float, help='beta')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--mode', default='baseline', type=str, help='baseline or labelemb?')
parser.add_argument('--data', default='./', type=str, help='file path of the dataset')
parser.add_argument('--num', default='0', type=str)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = ResNet18()
#torch.manual_seed(44)
if use_cuda:
    #torch.cuda.manual_seed(8)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


optimizer = optim.Adam(net.parameters())

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.2 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def my_loss(logit, prob):
    """ Cross-entropy function"""
    soft_logit = F.log_softmax(logit)
    loss = torch.sum(prob*soft_logit, 1)
    return loss

def comp_Loss(epoch, out1, out2, tar, emb_w, targets, mode):
    """Compute the total loss"""
    out2_prob = F.softmax(out2)
    tau2_prob = F.softmax(out2 / args.tau).detach()
    soft_tar = F.softmax(tar).detach()# + 0.5*F.softmax(tar1).detach()
    L_o1_y = F.cross_entropy(out1, targets)
    if mode == 'baseline':
        return L_o1_y

    alpha = args.alpha # default:0.9
    beta = args.beta # default:0.5#adjust_alpha(epoch)
    _, pred = torch.max(out2,1)
    mask = pred.eq(targets).float().detach()
    L_o1_emb = -torch.mean(my_loss(out1, soft_tar))

    L_o2_y = F.cross_entropy(out2, targets)
    L_emb_o2 = -torch.sum(my_loss(tar, tau2_prob)*mask)/(torch.sum(mask)+1e-8)
    gap = torch.gather(out2_prob, 1, targets.view(-1,1))-alpha
    L_re = torch.sum(F.relu(gap))
    #L2_loss = F.mse_loss(emb_w.t(), emb_w.detach())
    
    loss = beta*L_o1_y + (1-beta)*L_o1_emb +L_o2_y +L_emb_o2 +L_re
    return loss

# Training
print(args.mode)
if not os.path.isdir('100_results'):
            os.mkdir('100_results')
log = open('./100_results/'+args.num +args.mode+'.txt', 'a')

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets, requires_grad = False)
        outputs = net(inputs, targets, epoch, batch_idx)
        out1, out2, tar, emb_w = outputs

        loss = comp_Loss(epoch, out1, out2, tar, emb_w, targets, args.mode)
        loss.backward()
        adjust_learning_rate(optimizer, epoch)
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(out1.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    log.write(str(epoch)+' '+str(correct/total) +' ')
    if args.mode != 'baseline':# and args.mode != 'emb':
        pickle.dump(emb_w.data.cpu().numpy(), open('./100_results/'+args.num+args.mode+'embedding.pkl', 'wb'))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, _, _,_ = net(inputs, targets, -1,batch_idx)
        loss = F.cross_entropy(out, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    log.write(str(correct/total)+' '+str(test_loss) +'\n')
    log.flush()

    # Save checkpoint.
    '''
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    '''


for epoch in range(start_epoch, 100):
    train(epoch)
    test(epoch)

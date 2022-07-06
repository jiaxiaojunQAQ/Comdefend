import os, time
import argparse
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import shutil
import matplotlib
from PIL import Image

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import math
import torch._utils

# import celeba_data
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

from pdb import set_trace as st

import models.cifar as models


def save_checkpoint(state_dict, is_best, filepath):
    torch.save(state_dict, filepath)
    if is_best:
        shutil.copyfile(filepath, filepath.split('.pth')[0] + '_best.pth')


def save_image(save_path, tensor, ori_tensor):
    img = tensor.data.cpu().numpy()
    img = img.transpose(0, 2, 3, 1) * 255.0
    img = np.array(img).astype(np.uint8)
    img = np.concatenate(img, 1)

    ori_img = ori_tensor.data.cpu().numpy()
    ori_img = ori_img.transpose(0, 2, 3, 1) * 255.0
    ori_img = np.array(ori_img).astype(np.uint8)
    ori_img = np.concatenate(ori_img, 1)

    vis = np.concatenate(np.array([ori_img, img]), 0)
    img_pil = Image.fromarray(vis)
    # img_pil = img_pil.resize((w // 16, h // 16))
    img_pil.save(save_path)


class encoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 12, kernel_size=3, padding=1)
        )
        # Weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)

    def forward(self, x):
        return self.features(x)


class decoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(decoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # # Weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)

    def forward(self, x):
        return self.features(x)


def train(args, trainloader, enc, dec, optimizer):
    global mean, std
    enc.train()
    dec.train()
    acc = 0
    total = 0
    total_loss = 0
    count = 0
    flag = False
    for x, y in trainloader:
        x, y = Variable(x.cuda()), Variable(y.cuda())
        # x = torch.clamp(x, min=-1, max=1)
        # st()
        linear_code = enc(x)
        noisy_code = linear_code - torch.randn(linear_code.size()).cuda() * args.std
        binary_code = torch.sigmoid(noisy_code)
        recons_x = dec(binary_code)
        loss = ((recons_x - x) ** 2).mean() + (binary_code ** 2).mean() * 0.0001

        # linear_code = enc(x)
        # binary_code = torch.sigmoid(linear_code)
        # recons_x = dec(binary_code)
        # loss = ((recons_x - x)**2).mean()
        if not flag:
            flag = True
            img_tensor = x
            ori_img_tensor = recons_x

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if count % args.interval == 0:
            print("loss: {}".format(loss.item()))

        count += 1
    print("training averag loss: ", total_loss / float(count))

    return img_tensor, ori_img_tensor


def test(args, testloader, enc, dec, net):
    global mean, std
    enc.eval()
    dec.eval()
    acc = 0
    total = 0
    total_loss = 0
    count = 0
    flag = False
    img_tensor = False
    ori_img_tensor = False
    for x, y in testloader:
        x = Variable(x.cuda())
        y = Variable(y.cuda())
        linear_code = enc(x)
        noisy_code = linear_code - torch.randn(linear_code.size()).cuda() * args.std
        binary_code = torch.round(torch.sigmoid(noisy_code))
        # binary_code
        # z = x +x.sign().detach() - x.detach() differntiable version
        recons_x = dec(binary_code)
        if not flag:
            flag = True
            img_tensor = x
            ori_img_tensor = recons_x
        loss = ((recons_x - x) ** 2).mean()
        total_loss += loss.item()

        logits = net((recons_x - mean) / std)
        _, pred = torch.max(logits, dim=1)
        acc += (pred == y).sum().item()
        total += y.size()[0]
        count += 1
    acc = acc / float(total)
    avg_loss = total_loss / float(count)
    print("test averag loss: ", avg_loss, " test acc: ", acc)
    return acc, avg_loss, img_tensor, ori_img_tensor


def test_network_acc(testloader, net):
    global mean, std
    net.eval()
    acc = 0.0
    total = 0.0
    for x, y in testloader:
        x = x.cuda()
        y = y.cuda()
        logits = net((x - mean) / std)
        _, pred = torch.max(logits, dim=1)
        acc += (pred == y).sum().item()
        total += y.size()[0]
    print("accuracy: ", acc / float(total))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch comdefend Training')

    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--interval', default=100, type=int)
    parser.add_argument('--std', default=20.0, type=float)
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset', default='svhn', type=str)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--sigma', default=2, type=float)
    parser.add_argument('--workers', default=4, type=int)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    transform = transforms.Compose([
        # transforms.Scale(32),
        transforms.ToTensor()
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    global mean, std
    mean = torch.Tensor([0.5, 0.5, 0.5]).cuda()
    std = torch.Tensor([0.5, 0.5, 0.5]).cuda()
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)

    # if args.dataset == 'cifar':
    #     trainloader = torch.utils.data.DataLoader(
    #         datasets.CIFAR10('./data/cifar/', train=True, transform=transform, download=True),
    #         batch_size=args.batch_size, shuffle=True)

    #     testloader = torch.utils.data.DataLoader(
    #         datasets.CIFAR10('./data/cifar/', train=False, transform=transform, download=True),
    #         batch_size=args.batch_size, shuffle=False)

    #     net = models.__dict__['resnet'](num_classes=10,depth=50)
    #     checkpoint_file = "models_chaowei/svhn_checkpoint.pth"

    #     checkpoint = torch.load( "models_chaowei/svhn_checkpoint.pth")
    #     net.load_state_dict(checkpoint['model'])
    #     print("svhn")

    if args.dataset == 'svhn':
        trainloader = torch.utils.data.DataLoader(
            datasets.SVHN('./data/SVHN/', split='train', transform=transform, download=True),
            batch_size=args.batch_size, shuffle=True)

        testloader = torch.utils.data.DataLoader(
            datasets.SVHN('./data/SVHN/', split='test', transform=transform, download=True),
            batch_size=args.batch_size, shuffle=False)

        net = models.__dict__['resnet'](num_classes=10, depth=50)
        checkpoint_file = "save/svhn_checkpoint.pth"

        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['model'])
        print("svhn")
    else:
        raise NotImplementedError("should implement this!")

    enc = encoder()
    dec = decoder()

    # generator.cuda()
    enc = enc.cuda()
    dec = dec.cuda()
    net = net.cuda()
    test_network_acc(testloader, net)

    params = list(enc.parameters()) + list(dec.parameters())
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    best_acc = 0
    best_loss = 1e10
    isbest = False
    for epoch in range(args.max_epochs):
        train_img, train_ori = train(args, trainloader, enc, dec, optimizer)
        test_acc, test_loss, test_img, test_ori = test(args, testloader, enc, dec, net)

        state_dict = {"enc": enc.state_dict(), "dec": dec.state_dict(), "optimizer": optimizer, "test_acc": test_acc,
                      "args": args}
        filepath = "models_comdefend/{}_comdefend.pth".format(args.dataset)
        img_path = "models_comdefend/{}/train_{}.jpg".format(args.dataset, epoch)
        path = "/".join(img_path.split('/')[:-1])
        if not os.path.exists(path):
            os.makedirs(path)
        save_image(img_path, train_img, train_ori)
        img_path = "models_comdefend/{}/test_{}.jpg".format(args.dataset, epoch)
        save_image(img_path, test_img, test_ori)
        # if best_acc < test_acc:
        if best_loss > test_loss:
            best_acc = test_acc
            best_loss = test_loss
            isbest = True
        save_checkpoint(state_dict, isbest, filepath)
        print("best acc: {}, loss: {}".format(best_acc, best_loss))
        isbest = False

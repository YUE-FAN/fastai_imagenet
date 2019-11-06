"""
python eval_torch.py -a resnet50_1x1lap --gpu-id 0 -d /BS/yfan/nobackup/ILSVRC2012/val/ -j 32 --test-batch 100 --resume /BS/yfan/work/trained-models/dconv/checkpoints/imagenet/resnet501x1lap_90_lr0.1_bs256/resnet501x1lap_9942_90/model_best.pth.tar --layer 99
"""

from __future__ import print_function

import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.datasets as datasets
from vggfy import VGG16, VGG16_1d, VGG16_1x1LMP, VGG16_1x1LAP
from resnetfy import Resnet50, Resnet50_1d, Resnet152_1d, Resnet50_1x1, Resnet152_1x1, Resnet152_1x1LAP, \
    Resnet152_truncated, Resnet152_1x1LMP
from resnetfy import Resnet50_1x1LMP, Resnet50_1x1LAP, Resnet50_truncated
from mobilenetv2 import MobileNetV2_1x1LMP, MobileNetV2_1x1LAP
from mobilenet import MobileNetV1_1x1LMP, MobileNetV1_1x1LAP

from utils import Bar, AverageMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Datasets
parser.add_argument('-d', '--dataset', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--test-batch', default=256, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20', help='model architecture: ')
parser.add_argument('--layer', type=int)
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu-id', default=None, type=str,
                    help='GPU id to use.')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

best_acc = 0  # best test accuracy
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True
best_acc = 0  # best test accuracy


class Rotation(object):
    def __init__(self, degrees, resample=3, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, img):
        return F.rotate(img, self.degrees, self.resample, self.expand, self.center)


class Translation(object):
    def __init__(self, translate, resample=3, fillcolor=0):
        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "translate should be a list or tuple and it must be of length 2."
        for t in translate:
            if not (-1.0 <= t <= 1.0):
                raise ValueError("translation values should be between 0 and 1")
        self.translate = translate
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, img):
        img_size = img.size
        max_dx = self.translate[0] * img_size[0]
        translate = (np.round(-max_dx), 0)
        return F.affine(img, 0.0, translate, 1.0, 0.0, resample=self.resample, fillcolor=self.fillcolor)


class Center_block(object):
    def __init__(self, block_ratio, block_color=[0, 0, 0]):
        if not (0.0 <= block_ratio <= 1.0):
            raise ValueError("block ratio should be between 0 and 1")
        self.block_ratio = block_ratio
        self.block_color = block_color

    def __call__(self, img):
        img_size = np.array([img.size(1), img.size(2)])
        block_size = np.round(self.block_ratio * img_size).astype(np.int)
        loc = np.round(img_size / 2 - block_size / 2).astype(np.int)  # upper left corner
        img[0, loc[0]:loc[0] + block_size[0], loc[1]:loc[1] + block_size[1]] = self.block_color[0]
        img[1, loc[0]:loc[0] + block_size[0], loc[1]:loc[1] + block_size[1]] = self.block_color[1]
        img[2, loc[0]:loc[0] + block_size[0], loc[1]:loc[1] + block_size[1]] = self.block_color[2]
        return img


def main():
    global best_acc
    num_classes = 1000

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('resnet50'):
        model = Resnet50(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('resnet152_1x1'):
        model = Resnet152_1x1(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('resnet152_1x1lap'):
        model = Resnet152_1x1LAP(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('resnet152_1x1lmp'):
        model = Resnet152_1x1LMP(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('resnet50_1x1lap'):
        model = Resnet50_1x1LAP(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('resnet50_1x1lmp'):
        model = Resnet50_1x1LMP(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('resnet152_truncated'):
        model = Resnet152_truncated(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('resnet50_truncated'):
        model = Resnet50_truncated(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('vgg16'):
        model = VGG16(args.drop, num_classes, True)
    elif args.arch.endswith('vgg16_1d'):
        model = VGG16_1d(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('vgg16_1x1lmp'):
        model = VGG16_1x1LMP(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('vgg16_1x1lap'):
        model = VGG16_1x1LAP(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('d1_resnet50'):
        model = Resnet50_1d(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('resnet50_1x1'):
        model = Resnet50_1x1(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('d1_resnet152'):
        model = Resnet152_1d(args.drop, num_classes, True, args.layer)
    elif args.arch.endswith('mobilenetv1_1x1lmp'):
        model = MobileNetV1_1x1LMP(1 - 0.999, num_classes, True, args.layer)
    elif args.arch.endswith('mobilenetv1_1x1lap'):
        model = MobileNetV1_1x1LAP(1 - 0.999, num_classes, True, args.layer)
    elif args.arch.endswith('mobilenetv2_1x1lmp'):
        model = MobileNetV2_1x1LMP(num_classes, args.layer)
    elif args.arch.endswith('mobilenetv2_1x1lap'):
        model = MobileNetV2_1x1LAP(num_classes, args.layer)
    else:
        raise Exception('arch can only be vgg16 or resnet50!')

    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu_id)
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])

    print('==> Preparing dataset %s' % args.dataset)
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # TODO: remember to get rid of it
        # Rotation(-30),
        # Translation((-60./224., 0)),

        transforms.ToTensor(),
        # Center_block(0.5),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((124/255, 116/255, 104/255), (0.229, 0.224, 0.225)), # TODO: for meanvalue_background val
    ])

    # valdir = os.path.join(args.dataset, 'val')
    valdir = args.dataset
    testset = datasets.ImageFolder(valdir, transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    print(valdir)
    test_loss, test_acc = test(testloader, model, criterion, args)
    print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    return


def test(testloader, model, criterion, args):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        bar = Bar('Processing', max=len(testloader))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu_id is not None:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)


if __name__ == '__main__':
    main()

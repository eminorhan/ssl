import argparse
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils import train, validate, adjust_learning_rate

parser = argparse.ArgumentParser(description='ImageNet fine-tuning or linear classification')
parser.add_argument('--imgnet-basedir', default='/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/', type=str, help='path to ImageNet')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay (default: 0)')
parser.add_argument('--print-freq', default=5000, type=int, help='print frequency (default: 5000)')
parser.add_argument('--schedule', default=[27, 29], nargs='*', type=int, help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--n_out', default=20, type=int, help='output dim')
parser.add_argument('--freeze-trunk', default=False, action='store_true', help='freeze trunk?')
parser.add_argument('--frac-retained', default=1.0, type=float, help='fraction of tr data retained')
parser.add_argument('--strong-augment', default=False, action='store_true', help='use strong data augmentation')

def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting body to non-trainable'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.module.fc.parameters():
            param.requires_grad = True

def main():

    args = parser.parse_args()
    print(args)

    model = models.resnext101_32x8d(pretrained=False)
    model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_out, bias=True)

    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    # if resume from a pretrained model
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading model '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            if args.freeze_trunk:
                print('Freezing trunk.')
                set_parameter_requires_grad(model)  # freeze the trunk
            model.module.fc = torch.nn.Linear(in_features=2048, out_features=1000, bias=True).cuda()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.freeze_trunk:
            print('Freezing trunk.')
            set_parameter_requires_grad(model)  # freeze the trunk
            model.module.fc = torch.nn.Linear(in_features=2048, out_features=1000, bias=True).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # Save file name
    savefile_name = 'ImageNet_{}_{}_{}.tar'.format(args.resume, args.freeze_trunk, args.frac_retained)

    # Data loaders
    traindir = os.path.join(args.imgnet_basedir, 'train')
    valdir = os.path.join(args.imgnet_basedir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.strong_augment:
        # use the stronger TC augmentations for training the ImageNet control model
        from utils import GaussianBlur
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
                ])
        )
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        )

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))

    if args.frac_retained < 1.0:
        print('Fraction of train data retained:', args.frac_retained)

        import numpy as np
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        train_idx = indices[:int(args.frac_retained * num_train)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    else:
        print('Using all of train data')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    acc1_list = []
    val_acc1_list = []

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1 = train(train_loader, model, criterion, optimizer, epoch, args)
        acc1_list.append(acc1)

        # ... then validate
        val_acc1 = validate(val_loader, model, args)
        val_acc1_list.append(val_acc1)

    torch.save({'acc1_list': acc1_list,
                'val_acc1_list': val_acc1_list,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, savefile_name)

if __name__ == '__main__':
    main()
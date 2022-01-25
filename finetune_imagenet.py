import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import set_seed, adjust_learning_rate, train_default, validate, load_model

parser = argparse.ArgumentParser(description='ImageNet fine-tuning or linear classification')
parser.add_argument('--imgnet-basedir', default='/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/', type=str, help='path to ImageNet')
parser.add_argument('--model', default='resnext101_32x8d', choices=['resnext101_32x8d', 'resnext50_32x4d'], help='model architecture')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', default=1000, type=int, help='number of classes in downstream task')
parser.add_argument('--num-outs', default=1000, type=int, help='number of outputs in pretrained')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay (default: 0)')
parser.add_argument('--print-freq', default=5000, type=int, help='print frequency (default: 5000)')
parser.add_argument('--schedule', default=[25, 45], nargs='*', type=int, help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--model-dir', type=str, default='/misc/vlgscratch4/LakeGroup/emin/baby-vision/scripts/tc_models', help='directory holding pretrained models')
parser.add_argument('--model-name', type=str, default='TC-SAYAVAKEPICUT', choices=['TC-SAYAVAKEPICUT', 'DINO-SAYAVAKEPICUT', 'TC-SAY-resnext', 'DINO-SAY-resnext'], help='evaluated model')
parser.add_argument('--freeze-trunk', default=False, action='store_true', help='freeze trunk?')
parser.add_argument('--frac-retained', default=1.0, type=float, help='fraction of tr data retained')
parser.add_argument('--strong-augment', default=False, action='store_true', help='use strong data augmentation')
parser.add_argument('--seed', default=1, type=int, help='random seed')

def main():

    args = parser.parse_args()
    print(args)

    # set random seed
    set_seed(args.seed)

    model = load_model(args)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # Save file name
    savefile_name = 'ImageNet_{}_{}_{}.tar'.format(args.model_name, args.freeze_trunk, args.frac_retained)

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

    train_acc1_list = []
    val_acc1_list = []

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_acc1 = train_default(train_loader, model, criterion, optimizer, epoch, args)
        train_acc1_list.append(train_acc1)

        # ... then validate
        val_acc1 = validate(val_loader, model)
        val_acc1_list.append(val_acc1)

    torch.save({'train_acc1_list': train_acc1_list,
                'val_acc1_list': val_acc1_list,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, savefile_name)

if __name__ == '__main__':
    main()
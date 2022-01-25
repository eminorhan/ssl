import argparse
import os
import warnings
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils import train, set_seed, GaussianBlur
from multiroot_image_folder import MultirootImageFolder

parser = argparse.ArgumentParser(description='Temporal classification training with video data')
parser.add_argument('--data-dirs', nargs='+', help='list of paths to datasets')
parser.add_argument('--model', default='resnext101_32x8d', choices=['resnext101_32x8d', 'resnext50_32x4d'], help='model')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=15, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='starts training from this epoch')
parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay (default: 0)')
parser.add_argument('--print-freq', default=10000, type=int, help='print frequency (default: 10000)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--n_out', default=1000, type=int, help='number of output dimensions')
parser.add_argument('--augmentation', default=True, action='store_false', help='use data augmentation')
parser.add_argument('--subject', default='SAYAVAKEPICUT', choices=['SAYAVAKEPICUT', 'SAY', 'S', 'A', 'Y'], help='subject')
parser.add_argument('--cache-path', default='', type=str, help='Cache path if dataset is cached')
parser.add_argument('--class-fraction', default=1.0, type=float, help='retained class fraction')

def main():

    args = parser.parse_args()
    print(args)

    # set random seed
    set_seed(args.seed)

    model = torchvision.models.__dict__[args.model](pretrained=False, num_classes=args.n_out)
    model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_out, bias=True)

    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.lr = args.lr
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),  # this is to make sure same as train tc set up
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

    if args.cache_path and os.path.exists(args.cache_path):
        print("Loading training dataset from {}".format(args.cache_path))
        train_dataset = torch.load(args.cache_path)
        train_dataset.transform = train_transforms
    else:
        print("Building training dataset from scratch")
        train_dataset = MultirootImageFolder(args.data_dirs, args.class_fraction, train_transforms)
        torch.save(train_dataset, args.cache_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None
    )

    print('Dataset size:', len(train_dataset))
    print('Loader size:', len(train_loader))
    print('Number of classes:', len(train_dataset.classes))

    savefile_name = 'TC_{}_{}_{}_{}'.format(args.subject, args.model, args.class_fraction, args.seed)

    acc1_list = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        acc1 = train(train_loader, model, criterion, optimizer, epoch, args)
        acc1_list.append(acc1)
        torch.save({'acc1_list': acc1_list, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    '{}_epoch_{}.tar'.format(savefile_name, epoch))

if __name__ == '__main__':
    main()
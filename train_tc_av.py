import argparse
import os
import warnings
import torch
import torchvision
import torch.backends.cudnn as cudnn
from torch.nn import Identity, Sequential, Linear, ReLU
from utils import train_av, set_seed, load_dataloader
from models import AVModel

parser = argparse.ArgumentParser(description='Temporal classification training with AV data')
parser.add_argument('--vision-data-dirs', nargs='+', help='list of paths to vision datasets')
parser.add_argument('--audio-data-dirs', nargs='+', help='list of paths to audio datasets')
parser.add_argument('--audio_model', default='resnet18', choices=['resnext101_32x8d', 'resnext50_32x4d', 'resnext18'], help='audio model')
parser.add_argument('--vision_model', default='resnext50_32x4d', choices=['resnext101_32x8d', 'resnext50_32x4d', 'resnext18'], help='vision model')
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
parser.add_argument('--vision-augmentation', default=True, action='store_false', help='use data augmentation for visual inputs')
parser.add_argument('--audio-augmentation', default=False, action='store_true', help='use data augmentation for audio inputs')
parser.add_argument('--subject', default='SAYAVAKEPICUT', choices=['SAYAVAKEPICUT', 'SAY', 'S', 'A', 'Y'], help='subject')
parser.add_argument('--vision-data-cache', default='', type=str, help='Cache path for visual data')
parser.add_argument('--audio-data-cache', default='', type=str, help='Cache path for audio data')
parser.add_argument('--class-fraction', default=1.0, type=float, help='retained class fraction')

def main():
    
    args = parser.parse_args()
    print(args)

    # set random seed
    set_seed(args.seed)

    # set up audio encoder
    audio_model = torchvision.models.__dict__[args.audio_model](pretrained=False)
    audio_model.fc = Identity()

    # set up vision encoder
    vision_model = torchvision.models.__dict__[args.vision_model](pretrained=False)
    vision_model.fc = Identity()

    # set up projection head
    projection_head = Sequential(Linear(in_features=4096, out_features=2048, bias=True), 
                                ReLU(), 
                                Linear(in_features=2048, out_features=args.n_out, bias=True)
                                )

    av_model = AVModel(vision_model, audio_model, projection_head)

    # DataParallel will divide and allocate batch_size to all available GPUs
    av_model = torch.nn.DataParallel(av_model).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(av_model.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # resume from a prior checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(args.resume)
            checkpoint = torch.load(args.resume)
            av_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.lr = args.lr
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # prepare data loaders
    print('Loading visual data...')
    vision_loader = load_dataloader(args.vision_augmentation, args.vision_data_cache, args.vision_data_dirs, args)

    print('Loading audio data...')
    audio_loader = load_dataloader(args.audio_augmentation, args.audio_data_cache, args.audio_data_dirs, args)

    savefile_name = 'TC_AV_{}_{}_{}'.format(args.subject, args.class_fraction, args.seed)

    acc1_list = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        acc1 = train_av(vision_loader, audio_loader, av_model, criterion, optimizer, epoch, args)
        acc1_list.append(acc1)
        torch.save({'acc1_list': acc1_list, 
                    'model_state_dict': av_model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    '{}_epoch_{}.tar'.format(savefile_name, epoch))

if __name__ == '__main__':
    main()
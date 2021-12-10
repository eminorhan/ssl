# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
from PIL import ImageFilter
import random
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, RandomResizedCrop, RandomApply, ColorJitter, RandomGrayscale, RandomHorizontalFlip, ToTensor
from multiroot_image_folder import MultirootImageFolder


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting body to non-trainable'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.module.fc.parameters():
            param.requires_grad = True

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.2 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# TODO: Consolidate two train and validate functions below
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg.cpu().numpy()

def train_labeleds(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg.cpu().numpy()

def train_av(vision_loader, audio_loader, model, criterion, optimizer, epoch, args):
    # TODO: make sure vision loader and audio loader have the same size
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(vision_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, ((images, target, v_idx), (sounds, _, a_idx)) in enumerate(zip(vision_loader, audio_loader)):
        
        # measure data loading time
        data_time.update(time.time() - end)

        sounds = sounds.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images, sounds)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg.cpu().numpy()

def validate(val_loader, model):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    print('End of epoch validation: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg.cpu().numpy()

def validate_labeleds(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)

            preds = np.argmax(output.cpu().numpy(), axis=1)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].cpu().numpy()[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg, preds, target.cpu().numpy(), images.cpu().numpy()

def load_dataloader(augmentation, data_cache, data_dirs, args):
    """Prepare data loader for av"""

    # we can add other types of data augmentations here later
    if augmentation:
        transforms = Compose([
            RandomResizedCrop(256, scale=(0.09, 1.0), ratio=(1.0, 1.0)),
            RandomApply([ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
            ])
    else:
        transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
            ])

    if data_cache and os.path.exists(data_cache):
        print("Loading data from cache at {}".format(data_cache))
        dataset = torch.load(data_cache)
        dataset.transform = transforms
    else:
        print("Building training dataset from scratch")
        dataset = MultirootImageFolder(data_dirs, args.class_fraction, transforms)
        torch.save(dataset, data_cache)

    # this makes sure the same indices are used for both audio and vision batches
    g = torch.Generator()
    g.manual_seed(args.seed)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, generator=g, sampler=None)

    ## print some useful info about data
    print('Dataset size:', len(dataset))
    print('Loader size:', len(loader))
    print('Number of classes in dataset:', len(dataset.classes))

    return loader

def load_dataloaders_labeleds(datadir, args, train_frac=0.5):
    """prepare data loaders for labeled S"""

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = ImageFolder(datadir, transform=Compose([Resize(256), CenterCrop(224), ToTensor(), normalize]))
    test_data = ImageFolder(datadir, transform=Compose([Resize(256), CenterCrop(224), ToTensor(), normalize]))

    num_train = len(train_data)

    print('Total data size is', num_train)

    indices = list(range(num_train))
    split = int(np.floor(train_frac * num_train))
    np.random.shuffle(indices)

    if args.subsample:
        num_data = int(0.1 * num_train)
        train_idx, test_idx = indices[:(num_data // 2)], indices[(num_data // 2):num_data]
    else:
        train_idx, test_idx = indices[:split], indices[split:]

    print('Training data size is', len(train_idx))
    print('Test data size is', len(test_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    return trainloader, testloader

def load_model(args):
    """loads model (tidy up in due course)"""
    import torchvision.models as models
    num_classes = args.num_classes

    # model definitions
    # TODO: Maybe wrap around DP at the final stage (possibly inside the main finetune code)
    if args.model_name == 'random':
        model = models.resnext50_32x4d(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()       
        set_parameter_requires_grad(model)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        model = torch.nn.DataParallel(model).cuda()
    elif args.model_name == 'imagenet':
        model = models.resnext50_32x4d(pretrained=True)
        model = torch.nn.DataParallel(model).cuda()
        set_parameter_requires_grad(model)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        model = torch.nn.DataParallel(model).cuda()
    elif args.model_name == 'WSL':
        torch.hub.set_dir('/misc/vlgscratch4/LakeGroup/emin/robust_vision/pretrained_models')
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        set_parameter_requires_grad(model)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        model = torch.nn.DataParallel(model).cuda()  
    elif args.model_name.startswith('DINO'):
        from load_dino_model import load_dino_model
        # loads the pretrained DINO model (HACKY!)
        model_path = os.path.join(args.model_dir, args.model_name + '.pth')
        model = models.resnext50_32x4d(pretrained=False)
        model = load_dino_model(model, model_path, verbose=True)
        model = torch.nn.DataParallel(model).cuda()
        set_parameter_requires_grad(model)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        model = torch.nn.DataParallel(model).cuda()
    elif args.model_name.startswith('TC'):
        model_path = os.path.join(args.model_dir, args.model_name + '.tar')
        model = models.resnext50_32x4d(pretrained=False)
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.num_outs, bias=True)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        set_parameter_requires_grad(model)  # freeze the trunk
        model.module.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True).cuda()    
    elif args.model_name.startswith('TC_AV'):
        from models import AVModel
        # set up audio encoder
        audio_model = models.resnet18(pretrained=False)
        audio_model.fc = torch.nn.Identity()

        # set up vision encoder
        vision_model = models.resnext50_32x4d(pretrained=False)
        vision_model.fc = torch.nn.Identity()

        # set up projection head
        projection_head = torch.nn.Sequential(
                                    torch.nn.Linear(in_features=2048+512, out_features=2048, bias=True), 
                                    torch.nn.ReLU(), 
                                    torch.nn.Linear(in_features=2048, out_features=args.num_outs, bias=True)
                                    )

        av_model = AVModel(vision_model, audio_model, projection_head)

        # DataParallel will divide and allocate batch_size to all available GPUs
        av_model = torch.nn.DataParallel(av_model).cuda()

        model_path = os.path.join(args.model_dir, args.model_name + '.tar')
        checkpoint = torch.load(model_path)
        av_model.load_state_dict(checkpoint['model_state_dict'])
        model = av_model.module.vision_encoder
        model = torch.nn.DataParallel(model).cuda()
        set_parameter_requires_grad(model)  # freeze the trunk
        model.module.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True).cuda()    
    
    return model
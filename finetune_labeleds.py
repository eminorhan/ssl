import argparse
import torch
import torch.backends.cudnn as cudnn
from utils import load_dataloaders_labeleds, load_model, train_labeleds, validate_labeleds

parser = argparse.ArgumentParser(description='Finetuning or linear probing on labeled S')
parser.add_argument('--labeleds-dir', type=str, default='/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5/', help='path to labeled S dataset')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size (on all GPUs)')
parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay (default: 0)', dest='weight_decay')
parser.add_argument('--print-freq', default=100, type=int, help='print frequency (default: 100)')
parser.add_argument('--model-dir', type=str, default='/misc/vlgscratch4/LakeGroup/emin/baby-vision/scripts/tc_models', help='directory holding pretrained models')
parser.add_argument('--model-name', type=str, default='TC-SAY-resnext', 
                    choices=['TC-S-resnext', 'TC-A-resnext', 'TC-Y-resnext', 'TC-SAY-resnext', 'DINO-SAY-resnext', 'imagenet', 'random'], help='evaluated model')
parser.add_argument('--num-outs', default=16127, type=int, help='number of outputs in pretrained model')
parser.add_argument('--num-classes', default=26, type=int, help='number of classes in downstream classification task')
parser.add_argument('--subsample', default=True, action='store_false', help='subsample data?')

def main():
    args = parser.parse_args()
    print(args)

    model = load_model(args)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # save-file name
    savefile_name = 'LabeledS_{}_{}.tar'.format(args.model_name, args.subsample)

    # data loaders
    train_loader, test_loader = load_dataloaders_labeleds(args.labeleds_dir, args)
    acc1_list = []
    val_acc1_list = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        acc1 = train_labeleds(train_loader, model, criterion, optimizer, epoch, args)
        acc1_list.append(acc1)

    # validate at end of epoch
    val_acc1, preds, target, images = validate_labeleds(test_loader, model, args)
    val_acc1_list.append(val_acc1)

    torch.save({'acc1_list': acc1_list,
                'val_acc1_list': val_acc1_list,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'preds': preds,
                'target': target,
                'images': images
                }, savefile_name)

if __name__ == '__main__':
    main()
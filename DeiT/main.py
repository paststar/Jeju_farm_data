import os
import argparse
from datetime import datetime

import torch.optim as optim
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from torch.utils.data import random_split
from train import train_transformer
from datasets import *
import utils as utils
from models import *
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description="")
#parser.add_argument('-db_path', help='gpu number', type=str, default='../Dataset')
#parser.add_argument('-baseline_path', help='baseline path', type=str, default='AD_Baseline')
#parser.add_argument('-save_path', help='save path', type=str, default='Logs/save_test')

# amazon, webcam, dslr
#parser.add_argument('-source', help='source', type=str, default='amazon')
#parser.add_argument('-target', help='target', type=str, default='webcam')
parser.add_argument('-experiment_name', help='experiment_name', type=str, default='DeiT_S_finetuning')

parser.add_argument('-workers', default=4, type=int, help='dataloader workers')
parser.add_argument('-gpu', help='gpu number', type=str, default='0')

parser.add_argument('-epochs', default=200, type=int)
parser.add_argument('-batch_size', default=4, type=int)
parser.add_argument('-lr', default=0.01, type=float)
parser.add_argument('-l2_decay', default=0.0001, type=float)
parser.add_argument('-momentum', default=0.9, type=float)
parser.add_argument('-nesterov', default=False, type=bool)

#parser.add_argument('-domain_num', default=2, type=int, help='number of domains, if don`t use doamin classifier set -1')
#parser.add_argument('-mask', default=False, type=bool, help='token mask')


def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Use GPU(s): {} for training".format(args.gpu))

    save_dir = os.path.join("./logs", args.experiment_name+" "+datetime.now().strftime('_%m%d_%H%M_'))
    writer = SummaryWriter(save_dir)
    #if os.path.exists(save_dir):
        #raise NameError('model dir exists!')
    #os.makedirs(save_dir)
    logging = utils.init_log(save_dir)
    _print = logging.info
    _print("############### experiments setting ###############")
    _print(args.__dict__)
    _print("###################################################")
    path = '/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/'

    #num_classes, resnet_type = utils.get_data_info()

    data_transforms = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    snap_dataset = Snapshot(path+"스냅샷", data_transforms)
    rgb_dataset = RGBdrone(path+"RGB드론", data_transforms)

    torch.manual_seed(0)
    len_ice_dataset = len(snap_dataset)
    len_ice_train = int(0.8 * len_ice_dataset)
    len_ice_valid = len_ice_dataset - len_ice_train
    snap_train_dataset, snap_valid_dataset = random_split(snap_dataset, [len_ice_train, len_ice_valid])

    len_ice_dataset = len(rgb_dataset)
    len_ice_train = int(0.8 * len_ice_dataset)
    len_ice_valid = len_ice_dataset - len_ice_train
    rgb_train_dataset, rgb_valid_dataset = random_split(rgb_dataset, [len_ice_train, len_ice_valid])

    snap_train_loader = torchdata.DataLoader(snap_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    rgb_train_loader = torchdata.DataLoader(rgb_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    snap_valid_loader = torchdata.DataLoader(snap_valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    rgb_valid_loader = torchdata.DataLoader(rgb_valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)

    lr, l2_decay, momentum, nesterov = args.lr, args.l2_decay, args.momentum, args.nesterov
    transformer = deit_small_distilled_patch16_224(pretrained=True, num_classes=10).cuda()

    optimizer = optim.SGD(transformer.parameters(), lr=lr, momentum=momentum, weight_decay=l2_decay, nesterov=nesterov)

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    #losses = [classifier_criterion]
    #discriminator_criterion = JSD().cuda()
    losses = [classifier_criterion]

    best_acc = 0

    for epoch in range(args.epochs):
        _print('\n### epoch : {} ###'.format(epoch))
        train_transformer(args, transformer, snap_train_loader, rgb_train_loader, optimizer, losses,
                    epoch,_print,writer)
        best_acc = utils.evaluate(transformer, snap_valid_loader, rgb_valid_loader, epoch,best_acc,save_dir,_print,writer)
        writer.flush()
    writer.close()

if __name__ == "__main__":
    main()

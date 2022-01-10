import os
import argparse
import time
from datetime import datetime

import torch.optim as optim
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from torch.utils.data import random_split
from train import train_transformer
#from dataset import *
from segdataset import *
import utils as utils
from models import *
import albumentations as album
from torch.utils.data import Subset
from sklearn.metrics import jaccard_score as jsc

parser = argparse.ArgumentParser(description="")
#parser.add_argument('-db_path', help='gpu number', type=str, default='../Dataset')
#parser.add_argument('-baseline_path', help='baseline path', type=str, default='AD_Baseline')
#parser.add_argument('-save_path', help='save path', type=str, default='Logs/save_test')
parser.add_argument('-experiment_name', help='experiment_name', type=str, default='deeplabv3_finetuning')

parser.add_argument('-workers', default=4, type=int, help='dataloader workers')
parser.add_argument('-gpu', help='gpu number', type=str, default='0')

parser.add_argument('-epochs', default=200, type=int)
parser.add_argument('-val_step', default=500, type=int)

parser.add_argument('-batch_size', default=8, type=int)
parser.add_argument('-lr', default=0.01, type=float)
parser.add_argument('-l2_decay', default=0.0001, type=float)
parser.add_argument('-momentum', default=0.9, type=float)
parser.add_argument('-nesterov', default=False, type=bool)

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Use GPU(s): {} for training".format(args.gpu))

    save_dir = os.path.join("./logs", args.experiment_name+" "+datetime.now().strftime('_%m%d_%H%M'))
    writer = SummaryWriter(save_dir)
    logging = utils.init_log(save_dir)
    _print = logging.info
    _print("############### experiments setting ###############")
    _print(args.__dict__)
    _print("###################################################")
    path = '/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/'

    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256),
            A.RandomCrop(256, 256),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256),
            A.CenterCrop(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_dataset = SegmentationDataset(root='/home/bang/Desktop/jeju/dataset/제주 월동작물 자동탐지 드론 영상/Training',
                                  image_folder='Image1', mask_folder='Mask', transforms=train_transform)
    train_dataloder = torchdata.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, pin_memory=True)

    val_dataset = SegmentationDataset(root='/home/bang/Desktop/jeju/dataset/제주 월동작물 자동탐지 드론 영상/Validation',
                                        image_folder='Image', mask_folder='Mask', transforms=val_transform)
    val_dataset = Subset(val_dataset, range(0,len(val_dataset),150))

    val_dataloder = torchdata.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.workers, pin_memory=True)


    lr, l2_decay, momentum, nesterov = args.lr, args.l2_decay, args.momentum, args.nesterov

    model = createDeepLabv3(outputchannels=3).cuda()

    #criterion = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_decay, nesterov=nesterov)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    #val_step = args.val_step
    val_step = 500
    _print('len of train {}, len of val len {}'.format(len(train_dataloder),len(val_dataloder)))
    for epoch in range(args.epochs):
        model.train()
        _print('\n### epoch : {} ###'.format(epoch))
        start = time.time()
        total_loss = 0
        total_train = 0
        best_miou = 0
        for step, data in enumerate(train_dataloder):
            img, mask = data['image'].cuda(), data['mask'].long().cuda()
            outputs = model(img)
            #print(outputs['out'].type(),img.type(),mask.type())
            loss = criterion(outputs['out'], mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss
            total_train += mask.size()[0]

            if step>=val_step and step%val_step==0:
                _print("{}/{} train loss : {:.4f} train time : {:.2f}s".format(step//val_step,len(train_dataloder)//val_step,
                    total_loss/total_train, time.time() - start))
                writer.add_scalar('train/loss',total_loss/total_train, step//val_step)

                model.eval()
                start = time.time()
                miou = 0
                with torch.no_grad():
                    for val_data in val_dataloder:
                        img, mask = val_data['image'].cuda(), val_data['mask'].long().cuda()
                        outputs = model(img)
                        pred = torch.argmax(outputs['out'], dim=1)
                        iou = jsc(pred.cpu().numpy().reshape(-1), mask.cpu().numpy().reshape(-1), average=None)
                        miou += np.sum(iou)/len(iou)

                    _print("{}/{} val mIOU : {:.4f} val time : {:.2f}s".format(step // val_step,
                                len(train_dataloder) // val_step, miou / len(val_dataloder),time.time() - start))
                    writer.add_scalar('val/mIOU', miou / len(val_dataloder),step // val_step)

                writer.flush()

                if best_miou < miou:
                    best_miou = miou
                    _print("## best mIOU: {:.4f}, model is saved at {} steps of {} epochs ##".format(miou / len(val_dataloder), step, epoch))
                    torch.save(model.state_dict(), save_dir + '/' + 'best_deeplabV3.pt')

                total_loss = 0
                total_train = 0

                model.train()
                start = time.time()
        scheduler.step()
    writer.close()

if __name__ == "__main__":
    main()

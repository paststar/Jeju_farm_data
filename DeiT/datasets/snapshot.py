import glob
import cv2
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
import xml.etree.ElementTree as ET

change_id = {v:i for i, v in enumerate([1,2,3,4,6,11,12,13,14,15])}

data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return Image.fromarray(img)
    except IOError:
        print('Cannot load image ' + path)

class Snapshot(data.Dataset):
    def __init__(self, root, transform=None, loader = img_loader):
        self.root = root
        self.transform = transform
        self.image_list = glob.glob(root + "/*/학습데이터/image/*.jpg")
        self.loader = img_loader
        # self.xml_list = []
        #
        # for i in self.image_list:
        #     tmp = i
        #     tmp =tmp.replace('image','xml')
        #     tmp = tmp.replace('jpg','xml')
        #     self.xml_list.append(tmp)

    def __getitem__(self, index):
        img = self.loader(self.image_list[index])
        img = self.transform(img)
        # tree = ET.parse(self.xml_list[index])
        # root = tree.getroot()
        # lable = int(root.find('CROPS_ID').text)
        lable = change_id[int(self.image_list[index].split('.')[0][-2:])]

        return img,lable

    def __len__(self):
        return len(self.image_list)

if __name__ == "__main__":
    batch_size = 16
    workers = 4
    path = '/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/스냅샷'

    trainset = Snapshot(path,data_transforms['train'])
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                   num_workers=workers, pin_memory=True, drop_last=True)

    L = torch.tensor([])
    for step, src_data in enumerate(train_loader):
        tgt_imgs, tgt_labels = src_data
        L=torch.cat((L, tgt_labels))
    print(torch.unique(L))
    print(len(trainset))



import glob
import cv2
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import xml.etree.ElementTree as ET

if __name__ == "__main__":
    for ind, i in enumerate(glob.glob('/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/스냅샷/*/학습데이터/xml/*.xml')):
        tree = ET.parse(i)
        root = tree.getroot()
        lable = int(root.find('CROPS_ID').text)
        tmp = int(i.split('.')[0][-2:])

        if lable != tmp:
            print(lable,tmp, i)

    print(len(glob.glob('/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/스냅샷/*/학습데이터/xml/*')))


    # for j in range(20):
    #     for i in glob.glob('/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/스냅샷/*/학습데이터/xml/*.xml'):
    #         tree = ET.parse(i)
    #         root = tree.getroot()
    #         lable = int(root.find('CROPS_ID').text)
    #
    #         if lable == j:
    #             print(j," : ", i)
    #             break





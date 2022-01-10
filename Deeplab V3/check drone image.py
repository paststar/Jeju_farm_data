import os
import cv2

mask_path = '../dataset/제주 월동작물 자동탐지 드론 영상/Validation/Mask'
img_path = '../dataset/제주 월동작물 자동탐지 드론 영상/Validation/Image'
#path = '../dataset/제주 월동작물 자동탐지 드론 영상/Training/'

for i in os.listdir(img_path):
    mask = cv2.imread(os.path.join(mask_path,i))
    img = cv2.imread(os.path.join(img_path,i))

    cv2.imshow("img",img)
    cv2.imshow("mask", mask)
    cv2.imshow("color_mask", cv2.applyColorMap(mask*20, cv2.COLORMAP_JET))
    cv2.waitKey(0)



from osgeo import gdal
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import os

path = "/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/다중분광드론/15. 기장/학습데이터/image/*.tif"
#path ="/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/초분광드론/*/학습데이터/image/*.tif"

m = float("inf")
M = 0

for ind, i in enumerate(glob.glob(path)):
    # dataset = gdal.Open(os.path.join("/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/다중분광드론/15. 기장/학습데이터/image",
    #                                  "2107M0110_0216.tif"))
    img_name = i.split('/')[-1].rstrip(".tif")
    rgb_img_path = os.path.join('/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/RGB드론/1cm/15. 기장/image',
                 img_name.replace('M', 'D') + ".jpg")
    if os.path.isfile(rgb_img_path):
        rgb_img = cv2.imread(rgb_img_path)
        cv2.imshow("origin",rgb_img)
    else:
        #cv2.imshow("origin", np.zeros((512,512,3)))
        continue

    dataset = gdal.Open(i)

    # band1 = dataset.GetRasterBand(1)  # Red channel
    # band2 = dataset.GetRasterBand(2)  # Green channelv
    # band3 = dataset.GetRasterBand(3)  # Blue channel
    #
    # b1 = band1.ReadAsArray()
    # b2 = band2.ReadAsArray()
    # b3 = band3.ReadAsArray()
    #
    # img = np.dstack((b1, b2, b3))
    # img = cv2.convertScaleAbs(img, alpha=0.03)
    # cv2.imshow("rgb", img)
    # cv2.waitKey(0)

    res = np.zeros((300,1))
    for j in range(1,dataset.RasterCount+1):
        band = dataset.GetRasterBand(j)
        arr = band.ReadAsArray()
        arr = arr/255
        #img = cv2.resize(cv2.applyColorMap(arr.astype(np.uint8), cv2.COLORMA P_JET),(300,300))
        img = cv2.resize(arr.astype(np.uint8), (300, 300))
        res = np.concatenate((res,img),axis=1).astype(np.uint8)
        #cv2.imshow("gray"+str(j),img)
    cv2.imshow("res", res)
    cv2.imwrite("./result/res"+str(ind)+'.png', res)
    cv2.imwrite("./result/img"+str(ind)+'.png', rgb_img)
    cv2.waitKey(0)

        # plt.imshow(arr)
        # plt.waitforbuttonpress(1)
        # plt.close()

        # m = min(m,np.min(arr))
        # M = max(M,np.max(arr))

#print(m,M)
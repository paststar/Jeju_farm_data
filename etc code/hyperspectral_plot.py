from osgeo import gdal
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import os

#path = "/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/다중분광드론/15. 기장/학습데이터/image/*.tif"
path ="/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/초분광드론/02. 당근/학습데이터/image/*.tif"

for ind, i in enumerate(glob.glob(path)):
    # img_name = i.split('/')[-1].rstrip(".tif")
    # rgb_img_path = os.path.join('/home/bang/Desktop/jeju/dataset/제주주요작물_데이터/RGB드론/1cm/11. 옥수수/image',
    #              img_name.replace('H', 'D') + ".jpg")
    # if os.path.isfile(rgb_img_path):
    #     rgb_img = cv2.imread(rgb_img_path)
    #     cv2.imshow("origin",rgb_img)
    # else:
    #     print(img_name)
    #     print(rgb_img_path)
    #     #print()
    #     #cv2.imshow("origin", np.zeros((512,512,3)))
    #     continue

    dataset = gdal.Open(i)
    fig = plt.figure(figsize=(10, 10))
    columns = 5
    rows = 5

    for n in range(dataset.RasterCount//(rows*columns)):
        res = np.zeros((1, 751))
        for r in range(rows):
            tmp = np.zeros((150, 1))
            for c in range(columns):
                num = 1+n*rows*columns + r*columns + c
                band = dataset.GetRasterBand(num)
                arr = band.ReadAsArray()
                img = arr.astype(np.uint8)
                #img = cv2.applyColorMap(cv2.resize(img, (150, 150)),cv2.COLORMAP_HSV)
                img = cv2.resize(img, (150, 150))
                tmp = np.concatenate((tmp, img), axis=1).astype(np.uint8)
            #print(res.shape)
            #print(tmp.shape)
            res = np.concatenate((res, tmp), axis=0).astype(np.uint8)

        cv2.imshow('res_'+str(n), res)
    cv2.waitKey(0)
    # for j in range(1, columns * rows + 1):
    #     band = dataset.GetRasterBand(j)
    #     arr = band.ReadAsArray()
    #     #print(np.unique(arr))
    #     #img = cv2.cvtColor((arr).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    #     img = arr.astype(np.uint8)
    #     fig.add_subplot(rows, columns, j)
    #     plt.imshow(img)
    # plt.show()

    #res = np.zeros((300,1))
    # for j in range(1,dataset.RasterCount+1):
    #     band = dataset.GetRasterBand(j)
    #     arr = band.ReadAsArray()
    #     arr = arr/255
    #     #img = cv2.resize(cv2.applyColorMap(arr.astype(np.uint8), cv2.COLORMA P_JET),(300,300))
    #     img = cv2.resize(arr.astype(np.uint8), (300, 300))
    #     res = np.concatenate((res,img),axis=1).astype(np.uint8)
    #     #cv2.imshow("gray"+str(j),img)
    # cv2.imshow("res", res)
    # cv2.imwrite("./result/res"+str(ind)+'.png', res)
    # cv2.imwrite("./result/img"+str(ind)+'.png', rgb_img)
    # cv2.waitKey(0)

        # plt.imshow(arr)
        # plt.waitforbuttonpress(1)
        # plt.close()

        # m = min(m,np.min(arr))
        # M = max(M,np.max(arr))

#print(m,M)
import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from libtiff import TIFF

# for i in glob.glob("../dataset/제주주요작물_데이터/다중분광드론/01. \월동무/학습데이터/image/*.tif"):
#     print(i)
#     img = plt.imread(i)
#     print(img)
#     plt.imshow(img)
#     plt.show()

tif = TIFF.open('a.tif', mode='r')
image = tif.read_image()

print(image)
plt.imshow(image)
plt.show()

import cv2
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

import numpy

image = cv2.imread(r"./assets/dia_5k_10um_640.tif", flags=cv2.IMREAD_GRAYSCALE)

# image_1 = image[300, 700:1900]

# binarized_image = cv2.adaptiveThreshold(image_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, -22)
# ret, thresh3 = cv2.threshold(image_1, 68, 255, cv2.THRESH_BINARY)
# plt.figure()
# plt.plot(image, label='primary image')
# plt.plot(thresh3, label='binarized image')
# plt.legend()
# plt.show()

image_2 = image[300:1500, 700:1900]
binarized_image = cv2.adaptiveThreshold(image_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -10)
ret, thresh3 = cv2.threshold(image_2, 68, 255, cv2.THRESH_BINARY)

plt.figure()
plt.subplot(2, 1, 1)
cs = plt.contourf(image_2, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
plt.title("primary image")
cbar = plt.colorbar(cs)
plt.subplot(2, 1, 2)
plt.contourf(thresh3, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
plt.title("binarized image")
cbar = plt.colorbar(cs)
plt.legend()
plt.show()

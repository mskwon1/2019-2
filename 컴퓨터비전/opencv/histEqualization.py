import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('barbara.jpg', 0)
equalized_img = cv2.equalizeHist(img)

plt.figure(figsize=(14,7))

plt.subplot(121)
plt.imshow(img, cmap='gray')

plt.subplot(122)
plt.imshow(equalized_img, cmap='gray')

plt.show()

hist = cv2.calcHist([img],[0],None,[256],[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
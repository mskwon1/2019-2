import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('barbara.jpg', 0)
equalized_img = cv2.equalizeHist(img)

hist = cv2.calcHist([img], [0], None, [256], [0,256])

# y = 255 - min / (max - min) * x
stretched_img = (img - hist.min())/(hist.max()-hist.min())*255

# 새로운 피규어를 만듬, figsize (width, height) // 단위 = 인치
plt.figure(figsize=(14,7))

# 1행 3열 index(10이하)
plt.subplot(131)
plt.imshow(img, cmap='gray')

plt.subplot(132)
plt.imshow(equalized_img, cmap='gray')

plt.subplot(133)
plt.imshow(stretched_img, cmap='gray')

plt.show()
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
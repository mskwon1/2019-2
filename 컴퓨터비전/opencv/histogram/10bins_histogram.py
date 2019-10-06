import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogram_10bins(img) :
    bins = np.zeros(10, np.int32)

    height, width = img.shape

    for i in range(0, height):
        for j in range(0, width):
            bins[int(img[i][j]/26)] += 1

    return bins

color_img = cv2.imread('barbara.jpg') # 리턴값은 numpy의 ndarray 타입
RGB_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

R_bins = histogram_10bins(RGB_img[:,:,0])
G_bins = histogram_10bins(RGB_img[:,:,1])
B_bins = histogram_10bins(RGB_img[:,:,2])

# 일정한 간격으로 띄여진 ndarray 리턴
x = np.arange(len(R_bins))
width = 0.2
# scalar의 sequence, height, width, color
plt.bar(x - width, R_bins, width, color='r')
plt.bar(x, G_bins, width, color='g')
plt.bar(x + width, B_bins, width, color='b')
plt.show()
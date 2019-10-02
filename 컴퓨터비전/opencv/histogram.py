import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram(img):
    # bins = np.zeros(16, np.int32)
    bins = np.zeros(256, np.int32)

    width, height = img.shape

    for i in range(0, height):
        for j in range(0, width):
            # bins[int(img[i][j]/16)] += 1
            bins[img[i][j]] += 1

    return bins

def histogram_10bins(img):
    bins = np.zeros(10, np.int32)
    height, width = img.shape

    for i in range(0, height):
        for j in range(0, width):
            bins[int(img[i][j]/26)] += 1

    return bins

img = cv2.imread('barbara.jpg', 0)

# 히스토그램 함수 이용
# bins = histogram(img)
#
# plt.plot(bins)
# plt.xlim([0,255])
# # plt.xlim([0,15])
# plt.show()

# OpenCV 이용
## cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

# hist = cv2.calcHist([img],[0],None,[256],[0,256])
# plt.plot(hist)
# plt.xlim([0,256])
# plt.show()

# Color image histogram
# color_img = cv2.imread('barbara.jpg')
# RGB_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
#
# R_bins = histogram(RGB_img[:,:,0])
# G_bins = histogram(RGB_img[:,:,1])
# B_bins = histogram(RGB_img[:,:,2])
#
# plt.plot(R_bins, 'r-', G_bins, 'g-', B_bins, 'b-')
# plt.show()

#color image histogram 표현 (10_bins)
# color_img = cv2.imread('barbara.jpg')
# RGB_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
#
# R_bins = histogram_10bins(RGB_img[:,:,0])
# G_bins = histogram_10bins(RGB_img[:,:,1])
# B_bins = histogram_10bins(RGB_img[:,:,2])
#
# x = np.arange(len(R_bins))
# width = 0.2
# plt.bar(x - width, R_bins, width, color='r')
# plt.bar(x , G_bins, width, color='g')
# plt.bar(x + width, B_bins, width, color='b')
#
# plt.show()

# equalization
equalized_img = cv2.equalizeHist(img)

# stretched_img = (img - )

# plt.figure(figsize=(14,7))
#
# plt.subplot(121)
# plt.imshow(img, cmap='gray')
#
# plt.subplot(122)
# plt.imshow(equalized_img, cmap='gray')
#
# # plt.subplot(123)
# # plt.imshow(equalized_img, cmap='gray')
#
# plt.show()
#
# hist = cv2.calcHist([img],[0],None,[256],[0,256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max()/ cdf.max()

#histogram and CDF
hist = cv2.calcHist([img],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
CDF = hist_norm.cumsum()

#initialization
bins = np.arange(256)
fn_min = np.inf
thresh = -1

#Otsu algorithm operation
for i in range(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = CDF[i],CDF[255]-CDF[i] # cum sum of classes

    if q1 == 0:
        q1 = 0.00000001
    if q2 == 0:
        q2 = 0.00000001
    b1,b2 = np.hsplit(bins,[i]) # weights
    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i
# find otsu's threshold value with OpenCV function
ret, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print( "thresh: {} ret: {}".format(thresh, ret) )

binary_img = np.zeros((img.shape[1], img.shape[0]), np.uint8)

for i in range(0,img.shape[1]):
    for j in range(0,img.shape[0]):
        if img[i][j]<thresh:
            binary_img[i][j] = 0
        else:
            binary_img[i][j]  = 255;


plt.figure(figsize=(14,7))

plt.subplot(121)
plt.imshow(otsu,cmap='gray')

plt.subplot(122)
plt.imshow(binary_img,cmap='gray')

plt.show()
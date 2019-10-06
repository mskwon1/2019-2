import numpy as np
import cv2
import matplotlib.pyplot as plt

# histogram and CDF
img = cv2.imread('barbara.jpg', 0)
hist = cv2.calcHist([img], [0], None, [256], [0,256])
# ravel 함수 : numpy 배열을 1차원으로 바꿔주는 함수
hist_norm = hist.ravel()/hist.max() # 정규 히스토그램
CDF = hist_norm.cumsum()    # 해당 인덱스까지의 누적 합

# initialization
bins = np.arange(256)   # 0~255 배열 생성
fn_min = np.inf     # 양의 무한대 초기화
thresh = -1     # -1로 초기화

# Otsu algorithm operation
for i in range (1,256) :
    p1,p2 = np.hsplit(hist_norm, [i])   # probabilities
    q1,q2 = CDF[i], CDF[255] - CDF[i]   # cum sum of classes

    if q1 == 0:
        q1 = 0.00000001
    if q2 == 0:
        q2 = 0.00000001
    b1,b2 = np.hsplit(bins, [i]) # weights
    # finding means and variances
    m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1, v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2
    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min :
        fn_min = fn
        thresh = i

# find otsu's threshold value with OpenCV function
# img, thershold_value, value, flag
ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("thresh: {} ret : {}".format(thresh, ret))

binary_img = np.zeros((img.shape[1], img.shape[0]), np.uint8)

# 임계값보다 크면 해당 픽셀은 백, 작으면 흑
for i in range(0, img.shape[1]):
    for j in range(0, img.shape[0]):
        if img[i][j] < thresh:
            binary_img[i][j] = 0
        else :
            binary_img[i][j] = 255

plt.figure(figsize = (14,7))

plt.subplot(121)
plt.imshow(otsu, cmap = 'gray')

plt.subplot(122)
plt.imshow(binary_img, cmap = 'gray')

plt.show()
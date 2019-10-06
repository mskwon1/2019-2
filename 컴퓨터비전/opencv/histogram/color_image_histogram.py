import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram(img):
    # 크기가 정해져 있고 모든 값이 0인 배열 생성 (크기,자료형)
    bins = np.zeros(256, np.int32) # 0~255 사이의 명암값을 넣을 배열
    # bins = np.zeros(16, np.int32) # 0~15 사이의 명암값을 넣을 배열

    # image.shape : 3차원 행렬로 표현,
    # (y size,x size, 값이 몇개의 원소로 이루어져있는지 - RGB일시 3)
    height, width = img.shape

    for i in range(0, height) :
        for j in range(0, width) :
            bins[img[i][j]] += 1    # 해당 픽셀의 명암값 +1
            # bins[int(img[i][j]/16)] += 1

    bins = bins / (width * height) # h'(l) = h(l)/(M * N)

    return bins

# 파일, flag (-1 : alpha 채널까지 포함하여 읽기
#              0 : 그레이스케일로 읽기(중간단계로 많이 사용)
#              1 : 칼라 파일로 읽기(투명한 부분은 무시) DEFAULT
color_img = cv2.imread('barbara.jpg') # 리턴값은 numpy의 ndarray 타입
RGB_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

R_bins = histogram(RGB_img[:,:,0])
G_bins = histogram(RGB_img[:,:,1])
B_bins = histogram(RGB_img[:,:,2])

plt.plot(R_bins, 'r-', G_bins, 'g-', B_bins, 'b-')
plt.show()
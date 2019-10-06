import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def eight_connect(img) :
    height, width = img.shape

    # 라벨값을 저장할 마스크
    mask = [[0 for x in range(width)] for y in range(height)]

    # 그룹에 부여할 라벨
    current_label = 1

    # 전체 이미지 배열을 돌며 checkFour 수행
    for i in range(0, height):
        for j in range(0, width):
            # 라벨링이 됐으면 current_label값을 다음으로 갱신
            if (checkEight(i, j, img, mask, 0, current_label)):
                # print ("labeled", current_label)
                current_label += 1


    # 랜덤 색깔 부여, 이미 색깔이 부여된 label은 Dictionary에 기억해놓은 값을 사용
    color_dict = {}
    labeled_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(height):
        for j in range(width):
            label = mask[i][j]
            if label != 0:
                # 해당 라벨이 dictionary에 없으면 새로 생성
                if label not in color_dict:
                    r = random.randint(0,255)
                    g = random.randint(0,255)
                    b = random.randint(0,255)
                    color_tuple = (r,g,b)
                    color_dict[label] = color_tuple
                # dictionary 참조 색깔 부여
                for k in range(3):
                    labeled_img[i][j][k] = color_dict[label][k]

    return labeled_img

# 재귀적으로 8 연결성을 탐색하는 함수
def checkEight(i, j, img, mask, mode, label):
    # 해당 픽셀의 값이 255이고, 라벨링 돼있지 않을 때 진입
    if (img[i][j] == 255 and mask[i][j] == 0) :
        mask[i][j] = label
        if (mode != 1 and i != len(img)-1):
            checkEight(i+1,j, img, mask, 3, label)
        if (mode != 2 and j != (len(img[0])-1)):
            checkEight(i,j+1, img, mask, 4, label)
        if (mode != 3 and i != 0):
            checkEight(i-1, j, img, mask, 1, label)
        if (mode != 4 and j != 0):
            checkEight(i,j-1, img, mask, 2, label)
        if (mode != 5 and i != len(img)-1 and j != len(img[0])-1):
            checkEight(i+1, j+1, img, mask, 7, label)
        if (mode != 6 and i != len(img)-1 and j != 0):
            checkEight(i+1, j-1, img, mask, 8, label)
        if (mode != 7 and i != 0 and j != 0):
            checkEight(i-1, j-1, img, mask, 5, label)
        if (mode != 8 and i != 0 and j != len(img[0])-1):
            checkEight(i-1, j+1, img, mask, 6, label)
        return True

    # 라벨링이 되지 않았으면 False 리턴
    return False



img = cv2.imread('sample.png', 0)

# cv2.imshow('sample', img)
labeled_img = eight_connect(img)

## image 출력
plt.imshow(labeled_img)
plt.grid(None)
plt.xticks([])
plt.yticks([])
plt.show()
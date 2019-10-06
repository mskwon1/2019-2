import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def four_connect(img) :
    height, width = img.shape

    # 라벨값을 저장할 마스크
    mask = [[0 for x in range(width)] for y in range(height)]

    # 그룹에 부여할 라벨
    current_label = 1

    # 전체 이미지 배열을 돌며 checkFour 수행
    for i in range(0, height):
        for j in range(0, width):
            # 라벨링이 됐으면 current_label값을 다음으로 갱신
            if (checkFour(i, j, img, mask, 0, current_label)):
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

# 재귀적으로 상하좌우를 탐색하는 함수
## mode값 0 : 상하좌우 다 체크 / 1 : 하 제외 / 2 : 우 제외 / 3 : 상 제외 / 4 : 좌 제외
def checkFour(i, j, img, mask, mode, label):
    # 해당 픽셀의 값이 255이고, 라벨링 돼있지 않을 때 진입
    if (img[i][j] == 255 and mask[i][j] == 0) :
        mask[i][j] = label
        # 위쪽 체크, 현재 함수를 위쪽에서 호출했거나 끝 쪽인 경우 진입 X
        if (mode != 3 and i != 0):
            checkFour(i-1, j, img, mask, 1, label)
            # print (i, ",", j, "checked north")
        # 왼쪽 체크, 현재 함수를 왼쪽에서 호출했거나 끝 쪽인 경우 진입 X
        if (mode != 4 and j != 0):
            checkFour(i,j-1, img, mask, 2, label)
            # print (i, ",", j, "checked east")
        # 아래쪽 체크, 현재 함수를 아래쪽에서 호출했거나 끝 쪽인 경우 진입 X
        if (mode != 1 and i != len(img)-1):
            checkFour(i+1,j, img, mask, 3, label)
            # print (i, ",", j, "checked south")
        # 오른쪽 체크, 현재 함수를 오른쪽에서 호출했거나 끝 쪽인 경우 진입 X
        if (mode != 2 and j != (len(img[0])-1)):
            checkFour(i,j+1, img, mask, 4, label)
            # print (i, ",", j, "checked west")
        return True

    # 라벨링이 되지 않았으면 False 리턴
    return False



img = cv2.imread('sample.png', 0)

# cv2.imshow('sample', img)
labeled_img = four_connect(img)

## image 출력
plt.imshow(labeled_img)
plt.grid(None)
plt.xticks([])
plt.yticks([])
plt.show()
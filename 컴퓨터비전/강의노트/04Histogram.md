# 히스토그램

## 계산

- [0, L-1] 사이의 명암값 각각이 영상에 몇 번 나타나는지 표시

- 히스토그램 h와 정규화 히스토그램

  ![1568704913950](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568704913950.png)

  ![1568704932920](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568704932920.png)

## 용도

- 영상의 특성 파악

  ![1568704967541](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568704967541.png)

- 평활화 (Histogram Equalization)

  - 히스토그램을 평평하게 만들어 주는 연산

  - 명암의 동적 범위를 확장하여 영상의 품질을 향상시켜줌

  - 누적 히스토그램 c(.)를 매핑 함수로 사용

    ![1568705029481](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705029481.png)

  ![1568705044331](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705044331.png)

  ![1568705057923](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705057923.png)

  - 영상처리 연산은 분별력을 가지고 활용 여부 결정해야 함

    ![1568705087151](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705087151.png)

## 히스토그램 역투영과 얼굴 검출

- 히스토그램 역투영

  - 히스토그램을 매핑 함수로 사용하여, 화소 값을 신뢰도 값으로 변환

- 얼굴 검출 예 : 모델 얼굴과 2차원 히스토그램

  ![1568705140044](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705140044.png)

- 2차원 히스토그램

  ![1568705151440](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705151440.png)

- 얼굴 검출

  - 모델 얼굴에서 구한 히스토그램 h_m은 
    화소의 컬러 값을 얼굴에 해당하는 신뢰도 값으로 변환해줌

  - 실제로는 비율 히스토그램 h_r을 사용

    ![1568705197809](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705197809.png)

- 히스토그램 역투영 알고리즘

  ![1568705212098](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705212098.png)

- 히스토그램 역투영 결과

  - 얼굴 영역은 높은 신뢰도 값, 손 영역도 높은 값
  - 한계
    - 비슷한 색 분포를 갖는 다른 물체 구별 못함
    - 검출 대상이 여러 색 분포를 갖는 경우 오류 가능성
  - 장점 : 배경을 조정할 수 있는 상황에 적합 (이동과 회전에 불변, 가림(occlusion)에 강인)

  ![1568705288903](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705288903.png)

# 이진 영상

## 이진화와 오츄 알고리즘

- 이진화 : 명암 영상을 흑과 백만 가진 이진 영상으로 변환

  ![1568705343185](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705343185.png)

- 임계값 방법

  - 두 봉우리 사이의 계곡을 임계값 T로 결정

  - 자연 영상에서는 계곡 지점 결정이 어려움

    ![1568705373516](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705373516.png)

- 오츄 알고리즘 [Otsu79]

  - 이진화 했을 때 흑 그룹과 백 그룹 각각이 균일할수록 좋다는 원리에 근거

  - 균일성은 분산으로 측정 (분산이 작을수록 균일성 높음)

  - 분산의 가중치 합 V_within(.)을 목적 함수로 이용한 최적화 알고리즘

    ![1568705426827](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705426827.png)

  - t-1 번째의 계산 결과를 t번째에 활용하여 빠르게 계산

    ![1568705452690](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705452690.png)

    ![1568705464549](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705464549.png)

    ​	![1568705475449](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705475449.png)

## 연결 요소

- 화소의 모양과 연결성

  ![1568705497010](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705497010.png)

- 연결요소 번호 붙이기

  - 4-연결성과 8-연결성

    ![1568705529510](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705529510.png)

- 범람 채움

  - 스택 오버플로우 위험

    ![1568705546237](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705546237.png)

- 열 단위로 처리하는 알고리즘

  ![1568705613524](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568705613524.png)
## 기술자

- 특징의 성질을 기술해주고, 여러 개의 값으로 구성된 벡터 형태이므로 **특징 벡터**라 부름

### 특징 기술자의 조건

- 매칭이나 인식에 유용하기 위한 몇 가지 요구 조건

  - 높은 분별력

  - 다양한 변환에 불변

    - 기하 불변성과 광도 불변성
    - 변환에도 불구하고 같은(유사한) 값을 갖는 **특징 벡터 추출**해야 함

    ![image-20191120135155837](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120135155837.png)

  - 특징 벡터의 크기(차원)

    - 차원이 낮을수록 계산 빠름
    - 차원의 저주 : 차원이 너무 커지면 계산이 그만큼 느려진다

- 응용에 따라 공변과 불변을 선택해야 함

### 관심점을 위한 기술자

- 관심점을 어떻게 기술할 것인가?
  - 들여다볼 **윈도우의 크기**가 중요
  - 스케일 정보 없는 관심점 (예) 해리스 코너
    - 윈도우 크기를 결정하는데 쓸 정보가 없음
    - 스케일 불변성 불가능
  - 스케일 정보 있는 관심점 (예) SIFT, SURF
    - 스케일 $\sigma$에 따라 윈도우 크기 결정
    - 스케일 불변성 달성

#### SIFT 기술자

- 4장에서 검출한 SIFT 키포인트(관심점)

  - 검출된 옥타브 $o$, 옥타브 내의 스케일 $\sigma_o$, 그 옥타브 영상에서 위치 $(r,c)$ 정보를 가짐

- SIFT 기술자의 **불변성**

  - 스케일 불변 달성
    - 윈도우를 옥타브 $o$, 옥타브 내의 스케일 $\sigma_o$, 그 옥타브 영상에서 위치 $(r,c)$에 씌움
  - 회전 불변 달성
    - 지배적인 방향 계산(윈도우 내의 그레이디언트 방향 히스토그램을 구한 후, 최대값을 갖는 방향 찾음)
    - 윈도우를 이 방향으로 씌움
  - 광도 불변 달성
    - 특징 벡터 $x$를 $||x||$로 나누어 정규화함

- SIFT 기술자 추출 알고리즘 [Lowe2004]

  - 윈도우를 4\*4의 16개의 블록으로 분할
    - 각 블록은 그레이디언트 방향 히스토그램 구함
    - 그레이디언트 방향은 8개로 양자화
  - 4\*4\*8 = 128차원 특징 벡터 $\bold{x}$
  - 화소 값은 보간으로 구함

  ![image-20191120140204742](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120140204742.png)

#### SIFT의 변형

- PCA-SIFT [Ke2004]

  - 키포인트에 39*39 윈도우 씌우고 도함수 $d_y$와 $d_x$계산
  - 39\*39*2차원의 벡터를 PCA를 이용하여 20차원으로 축소

- GLOH [Mikolajczyk2005a]

  - 원형 윈도우로 17*16차원의 벡터를 추출하고, PCA를 이용하여 128차원으로 축소

- 모양 콘텍스트 [Mikolajczyk2005a]

  - 원형 윈도우로 36차원 특징 벡터 추출

    <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120140739938.png" alt="image-20191120140739938" style="zoom: 67%;" />

#### 이진 기술자

- 빠른 매칭을 위해 특징 벡터를 이진열로 표현

  - 비교 쌍의 대소 관계에 따라 0 또는 1
  - 비교 쌍을 구성하는 방식에 따라 여러 변형

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120140816817.png" alt="image-20191120140816817" style="zoom:67%;" />

  - 매칭은 **해밍 거리**를 이용하여 빠르게 수행

- BRIEF, ORB, BRISK

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120140846385.png" alt="image-20191120140846385" style="zoom:67%;" />

### 영역 기술자

- 영역의 표현

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120140907743.png" alt="image-20191120140907743" style="zoom:67%;" />

#### 모멘트

- $(q+p)$차 모멘트

  - 물리에서 힘을 측정하는데 쓰는 모멘트를 영상에서 특징을 추출하는 데 적용

    <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120141224827.png" alt="image-20191120141224827" style="zoom:67%;" />

- 중심 모멘트

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120141236988.png" alt="image-20191120141236988" style="zoom:67%;" />

- 크기(스케일) 불변 모멘트

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120141306188.png" alt="image-20191120141306188" style="zoom: 80%;" />

- 회전 불변 모멘트 [Hu62]

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120141329707.png" alt="image-20191120141329707" style="zoom: 80%;" />

- 명암 영역의 모멘트

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120141404562.png" alt="image-20191120141404562" style="zoom: 67%;" />

  - 식 (6.8)의 크기 불변과 식 (6.9)의 회전 불변한 모멘트는 동일하게 정의됨
  - $f(y,x)$ : 명암값

- 영역의 중심이 어딨는지, x축/y축 방향으로 얼마나 산포되었는지, 평균과 분산이 어떤지

#### 모양

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120142009278.png" alt="image-20191120142009278" style="zoom: 67%;" />

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120142035753.png" alt="image-20191120142035753" style="zoom:67%;" />

- 투영

  - 데이터의 차원을 줄여보자

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120142059936.png" alt="image-20191120142059936" style="zoom:67%;" />

- 프로파일

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120142651740.png" alt="image-20191120142651740" style="zoom:67%;" />

  

#### 푸리에 기술자

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120142711548.png" alt="image-20191120142711548" style="zoom:67%;" />

- 신호를 기저 함수의 선형 결합으로 표현

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120143605902.png" alt="image-20191120143605902" style="zoom:67%;" />

  - 두 신호의 **계수** (0.5, 2.0)과 (2.0, 0.5)는 둘을 구별해주는 좋은 특징 벡터

- 신호가 입력되면 어떻게 계수를 알아낼 것인가?

  - 푸리에 변환으로 가능

    - $t(.)$가 계수에 해당
    - $i$축 : 공간 도메인, $u$축 : 주파수 도메인

    <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120143659172.png" alt="image-20191120143659172" style="zoom:67%;" />

- 영역에 푸리에 변환을 어떻게 적용하나?

  - 영역을 경계 표현으로 바꾼 뒤, 점의 위치를 복소수로 표현하면 푸리에 변환 식에 대입 가능

    <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120143824758.png" alt="image-20191120143824758" style="zoom:67%;" />

##### 신호를 표현할 수 있는 장법

- 테일러 급수

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191120143912346.png" alt="image-20191120143912346" style="zoom:67%;" />

  - Polynomials are not the best - unstable and not very physically meaningful

  - Easier to talk about "signals" in terms of its "frequencies"

    (how fast/often signals change, etc).

##### Jean Baptiste Joseph Fourier (1768 - 1830)

- Had Crazy Idea(1807)
  - Any periodic function can be rewritten as a weighted sum of Sines and Cosines of different frequencies
- Fourier Series
  - Possibly the greatest tool used in Engineering



### 텍스쳐

### 주성분 분석

### 얼굴 인식 : 고유 얼굴
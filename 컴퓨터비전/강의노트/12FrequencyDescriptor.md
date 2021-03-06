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
  
  - Any periodic function can be rewritten as a **weighted sum of Sines and Cosines of different frequencies**
  
- Fourier Series
  
  - Possibly the greatest tool used in Engineering
  
- A sum of sinusoids

  - Our building block

    <img src="../../typora_images/12FrequencyDescriptor/image-20191125134710417.png" alt="image-20191125134710417" style="zoom:50%;" />

    - Add enough of them to get any signal $f(x)$ you want
    
    - How many degrees of freedom?
    
    - What does each control?
    
    - Which one encodes the coarse vs fine sturcture of the signal
    
      <img src="../../typora_images/12FrequencyDescriptor/image-20191125134815624.png" alt="image-20191125134815624" style="zoom:67%;" />

- Fourier Transform

  - we want to understand the frequency $\omega$ of our signal. So let's reaprametrize the signal by $\omega$ instead of $x$

    <img src="../../typora_images/12FrequencyDescriptor/image-20191125135522457.png" alt="image-20191125135522457" style="zoom:67%;" />
  
    
    
    <img src="../../typora_images/12FrequencyDescriptor/image-20191125134518703.png" alt="image-20191125134518703" style="zoom: 50%;" />
    
    - 손실이 없는 압축이다
  
  <img src="../../typora_images/12FrequencyDescriptor/image-20191125135619920.png" alt="image-20191125135619920" style="zoom:67%;" />
  
  
  
  - Example
  
    <img src="../../typora_images/12FrequencyDescriptor/image-20191125140529831.png" alt="image-20191125140529831" style="zoom:67%;" />
  
   - Frequency Spectra
  
     <img src="../../typora_images/12FrequencyDescriptor/image-20191125140606922.png" alt="image-20191125140606922" style="zoom:67%;" />
  
     <img src="../../typora_images/12FrequencyDescriptor/image-20191125140617243.png" alt="image-20191125140617243" style="zoom:67%;" />
  
  - Fourier Transform Pairs
  
    <img src="../../typora_images/12FrequencyDescriptor/image-20191125140742633.png" alt="image-20191125140742633" style="zoom:67%;" />
  
  - Fourier Transform and Convolution
  
    <img src="../../typora_images/12FrequencyDescriptor/image-20191125140805359.png" alt="image-20191125140805359" style="zoom:50%;" />
  
    <img src="../../typora_images/12FrequencyDescriptor/image-20191125140828737.png" alt="image-20191125140828737" style="zoom: 50%;" />
  
    - 공간도메인에서의 컨볼루션 = 주파수도메인에서의 곱셈
  
  - Properties of Fourier Transform
  
    <img src="../../typora_images/12FrequencyDescriptor/image-20191125141336908.png" alt="image-20191125141336908" style="zoom:50%;" />
  
  - Example use : smoothing / blurring
  
    <img src="../../typora_images/12FrequencyDescriptor/image-20191125141420402.png" alt="image-20191125141420402" style="zoom:50%;" />
  
  - Low-pass filtering(Edge에 해당되는 부분이 날라가고 flat한 부분만 남아있음)
  
    <img src="../../typora_images/12FrequencyDescriptor/image-20191125141503816.png" alt="image-20191125141503816" style="zoom:50%;" />
  
    - 명암차이가 많으면 High frequency, 적으면 Low frequency
    - **오히려 얼굴인식이 잘됨**
  
  - High-pass filtering(Edge, Corner만 남음 - 명암차이가 큰 부분) 
  
    <img src="../../typora_images/12FrequencyDescriptor/image-20191125141639981.png" alt="image-20191125141639981" style="zoom:50%;" />

### 텍스쳐

- 텍스쳐

  - 일정한 패턴의 반복
  - 구조적 방법과 **통계적** 방법

  <img src="../../typora_images/12FrequencyDescriptor/image-20191125141924306.png" alt="image-20191125141924306" style="zoom: 67%;" />

#### 전역 기술자

- 에지 기반

  <img src="../../typora_images/12FrequencyDescriptor/image-20191125142000974.png" alt="image-20191125142000974" style="zoom:67%;" />

- 명암 히스토그램 기반

  <img src="../../typora_images/12FrequencyDescriptor/image-20191125142013923.png" alt="image-20191125142013923" style="zoom:67%;" />

  - uniform : 명암차이 변화가 가장 적을 때
  - entropy : 명암차이 변화가 가장 클 때(uniform과 상반되는 개념)

- 한계

  - 지역적인 정보 반영하지 못함

    <img src="../../typora_images/12FrequencyDescriptor/image-20191125142036922.png" alt="image-20191125142036922" style="zoom:67%;" />

#### 지역 관계 기술자

- 원리

  - 화소 사이의 이웃 관계를 규정하고, 그들이 형성하는 패턴을 표현

- 동시 발생 행렬

  - 이웃 관계를 이루는 화소 쌍의 명암이 $(j,i)$인 빈도수 세어, 행렬 $O$의 요소 $o_{ji}$에 기록

    <img src="../../typora_images/12FrequencyDescriptor/image-20191125142837919.png" alt="image-20191125142837919" style="zoom:67%;" />

  - 특징추출

    <img src="../../typora_images/12FrequencyDescriptor/image-20191125142946539.png" alt="image-20191125142946539" style="zoom: 80%;" />

- 지역 이진 패턴 (LBP) [Ojala96]

  - 8개 이웃과 대소관계에 따라 이진열을 만든 후 [0,255] 사이의 십진수로 변환

  - 모든 화소를 가지고 히스토그램 구성

    <img src="../../typora_images/12FrequencyDescriptor/image-20191125143031440.png" alt="image-20191125143031440" style="zoom:67%;" />

    - 기준값을 기준으로 0/1로 표현, 2진수로 바꾸고 10진수로 변환
    - 차원을 줄이는 효과

- 지역 삼진 패턴 (LTP)

  - 화소 값이 $p$라면, $p-t$보다 작으면 -1, $p+t$보다 크면 1, $[p-t, p+t]$ 사이면 0을 부여

  - 두 개의 LBP로 분리

  - 모든 화소를 가지고 히스토그램 구성

    <img src="../../typora_images/12FrequencyDescriptor/image-20191125143546881.png" alt="image-20191125143546881" style="zoom:67%;" />

- LBP와 LTP의 확장

  - 조명 변환에 불변이나, 8이웃만 보면 스케일 변화에 대처 하지 못함

  - 다양한 이웃을 이용한 스케일 불변 달성

    <img src="../../typora_images/12FrequencyDescriptor/image-20191125143837564.png" alt="image-20191125143837564" style="zoom:67%;" />

- LBP와 LTP의 응용

  - 얼굴 검출, 사람 검출, 자연 영상에서 글자 추출 등

### 주성분 분석

- 고차원 벡터를 저차원으로 축소

  - 정보 손실을 최소화하는 조건

    <img src="../../typora_images/12FrequencyDescriptor/image-20191125144203446.png" alt="image-20191125144203446" style="zoom:67%;" />

#### 원리

- 학습 집합 $X = \{x_1, x_2, x_3, ..., x_n\}$로 변환 행렬 $\bold{U}$를 추정

- $\bold{U}$는 $d * D$로서 $D$차원의 $x$를 $d$차원의 $y$로 변환

  <img src="../../typora_images/12FrequencyDescriptor/image-20191125144355397.png" alt="image-20191125144355397" style="zoom:67%;" />

  <img src="../../typora_images/12FrequencyDescriptor/image-20191125144347105.png" alt="image-20191125144347105" style="zoom:67%;" />

- 차원 축소를 어떻게 표현하나?

  - 축 $\bold{u}$상으로 투영으로 표현 $\hat{x} = \bold{ux}^T$
  - 그림 6-22는 2차원을 1차원으로 축소하는 상황

- 정보 손실을 어떻게 표현하나?

  - 정보란? 점들 사이의 거리나 상대적인 위치 등

  - 어느 것의 정보 손실이 최소인가? :arrow_forward: 직관적으로 판단하여 맨 오른쪽

    <img src="../../typora_images/12FrequencyDescriptor/image-20191125144515971.png" alt="image-20191125144515971" style="zoom:67%;" />
  
- PCA의 정보 손실 표현

  - 원래 공간에 퍼져 있는 정도를 변환된 공간이 얼마나 잘 유지하는지 측정
  - 이 수치를 변환된 공간에서 **분산**으로 측정

- 최적화 문제

  <img src="../../typora_images/12FrequencyDescriptor/image-20191129171933598.png" alt="image-20191129171933598" style="zoom:67%;" />

  <img src="../../typora_images/12FrequencyDescriptor/image-20191129171946781.png" alt="image-20191129171946781" style="zoom:67%;" />

#### 알고리즘

- 최대화 문제

  <img src="../../typora_images/12FrequencyDescriptor/image-20191129172006284.png" alt="image-20191129172006284" style="zoom:67%;" />

- u가 단위 벡터라는 조건을 포함시키면,

  <img src="../../typora_images/12FrequencyDescriptor/image-20191129172017557.png" alt="image-20191129172017557" style="zoom:67%;" />

- 도함수를 구하고, 도함수를 0으로 두고 정리하면,

  <img src="../../typora_images/12FrequencyDescriptor/image-20191129172038417.png" alt="image-20191129172038417" style="zoom:80%;" />

- 식 (6.39)의 의미

  <img src="../../typora_images/12FrequencyDescriptor/image-20191129172049915.png" alt="image-20191129172049915" style="zoom:67%;" />

<img src="../../typora_images/12FrequencyDescriptor/image-20191129172104029.png" alt="image-20191129172104029" style="zoom:80%;" />

- $D$차원을 $d$차원으로 축소

  - 지금까지는 $D$차원을 1차원으로 축소함

  - 공분산 행렬 $\sum$은 $D*D$이므로, $D$개의 고유 벡터가 있음

    - 이들은 서로 수직인 단위 벡터, 즉 $\bold{u}_j\bold{u}_i = 1$이고 $\bold{u}_j\bold{u}_i = 0, i\ne j$

  - 고유값이 큰 순서대로 상위 $d$개의 고유 벡터 $\bold{u}_1, \bold{u}_2, ..., \bold{u}_d$를 선택하고 식 (6.40)에 배치
    ($\bold{U}는 d*D$)

    <img src="../../typora_images/12FrequencyDescriptor/image-20191129172335637.png" alt="image-20191129172335637" style="zoom:67%;" />

- U를 이용한 차원 축소

  <img src="../../typora_images/12FrequencyDescriptor/image-20191129172347619.png" alt="image-20191129172347619" style="zoom:67%;" />

<img src="../../typora_images/12FrequencyDescriptor/image-20191129172355024.png" alt="image-20191129172355024" style="zoom:67%;" />

<img src="../../typora_images/12FrequencyDescriptor/image-20191129172401115.png" alt="image-20191129172401115" style="zoom:67%;" />

### 얼굴 인식 : 고유 얼굴

- 컴퓨터 비전에서 PCA의 응용 사례

  - 기술자 추출 : PCA-SIFT, GLOH 등
  - 가장 혁신적인 응용 :arrow_forward: 얼굴 인식

- 평균 얼굴

  <img src="../../typora_images/12FrequencyDescriptor/image-20191129172442760.png" alt="image-20191129172442760" style="zoom:67%;" />

- 얼굴 영상에 PCA 적용

  - 영상 $f_i$를 벡터 형태로 변환(벡터의 차원 $D=MN$) : 행 우선으로 재배치

    <img src="../../typora_images/12FrequencyDescriptor/image-20191129172538925.png" alt="image-20191129172538925" style="zoom:67%;" />

  - $n$개의 얼굴 영상으로 구성된 학습 집합 $X=\{x_1, x_2,..., x_n\}$을 입력으로 5절의 PCA를 적용

  - 이렇게 얻은 고유 벡터 $\bold{u_1}, \bold{u_2}, ..., \bold{u_d}$를 고유 얼굴이라 부름

  - 이들에 (6.43)을 역으로 적용하여 영상 형태로 바꾸면, 그림 6.25

    <img src="../../typora_images/12FrequencyDescriptor/image-20191129172702905.png" alt="image-20191129172702905" style="zoom:67%;" />

- 고유 얼굴의 활용 : 얼굴 영상 압축

  <img src="../../typora_images/12FrequencyDescriptor/image-20191129172729612.png" alt="image-20191129172729612" style="zoom:67%;" />

  - 역 변환으로 복원 가능

    <img src="../../typora_images/12FrequencyDescriptor/image-20191129172739173.png" alt="image-20191129172739173" style="zoom:67%;" />

- 고유 얼굴의 활용 : 얼굴 인식

  - PCA로 변환한 벡터 $\bold{y_i}$를 모델로 사용 : $Y = \{\bold{y_1},\bold{y_2},...,\bold{y_n}\}$
  - 테스트 영상 $f$가 입력되면 PCA로 $\bold{y}$를 구한 후, $Y$에서 가장 가까운 벡터를 찾아 그 부류로 분류

- 고유 얼굴 활용 시 주의점

  - 얼굴을 찍은 각도와 얼굴 크기, 영상 안에서의 얼굴 위치, 조명이 어느 정도 일정해야 함
  - 영상마다 다르고 그 변화가 클수록 성능이 떨어짐
  - Turk와 Pentrland의 연구 결과
    - 조명에 변화를 준 경우 96%, 각도에 변화를 준 경우 85%, 크기에 변화를 준 경우 64%의 정인식률을 얻음

  <img src="../../typora_images/12FrequencyDescriptor/image-20191129173014036.png" alt="image-20191129173014036" style="zoom:67%;" />


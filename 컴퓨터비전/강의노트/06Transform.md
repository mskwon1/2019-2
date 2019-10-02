## 기하 연산

- 일정한 기하 연산으로 결정된 화소의 명암값에 따라 새로운 값 결정

- Common Transformations

  ![1569991472601](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569991472601.png)
  - Affine : 직선에 대한 Parallel한 정도는 유지
  - Perspective : 하나의 소실점으로 모여지도록 하는 것, Parallel한 정보도 사라짐(가장 복잡)

- 2D image transformations (reference table)

  ![1569991579957](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569991579957.png)

  - rigid body object : 이동을 하거나 변환을 해도 모양이 바뀌지 않는 object
    - non-rigid body object : 이동하거나 변환하면 모양 바뀌는 object
    - 회전은 했지만 길이는 유지
  - similarity : rotation + translation, angle값은 유지
  - affine : rotation, translation, scale, angle 모두 바뀜, parallel 함은 유지
  - projective : parallel도 깨질 수 있음
    - 어느 시점에서 본

### Scaling

- **scaling** a coordinate : multiplying each of its components by a scalar

  - **unifrom scaling** : this scalar is the same for all components

    ![1569991966828](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569991966828.png)

  - **non-uniform scaling** : different scalars per component

    ![1569991978063](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569991978063.png)

- operation : x' = ax, y' = by

- matrix form : 

  ![1569992019618](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569992019618.png)

### 2-D Rotation

![1569992040518](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569992040518.png)

- Matrix Form

  ![1569992102651](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569992102651.png)

- sin, cos 연산이 비선형 함수이지만

  - **x' is a linear combination of x and y**
  - **y' is a linear combination of x and y**

- Inverse Transformation
  - Rotation by –theta
  - Rotation Matrices : ![1569992276409](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569992276409.png)

#### Basic Transformations

![1569992453082](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569992453082.png)

- Shear(기울임)는영상을 눕히는 셈,  x는 y에 변환을, y는 x에 변환을 줌
- Affine은 translation, scale, rotation, shear의 조합 중 
- 2x3 매트릭스중 앞의 2x2 매트릭스는 I/R/sR, 뒤 2x1 매트릭스가 translation 표현

#### Affine Transformations

![1569992753559](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569992753559.png)

- Affine Transformations : Combinations of linear transformations and translations
- Properties
  - Lines map to lines
  - Parallel lines remain parallel
  - Ratios are preserved
  - Closed under composition

- 3x3 행렬은 Honogenous transformation matrix
  - 단순히 차원을 하나 늘리는 것
  - 밑에 있는 값들이 0 0 1 고정

#### Projective Transformations

![1569992855856](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569992855856.png)

- Combos of affine transformations and projective warps

- Properties
  - Lines map to lines
  - Parallel lines do not necessarily remain parallel
  - Ratios are not preserved
  - Closed under composition
  - Models change of basis
  - Projective Matrix is defined up to a scale (8 DOF)
    - 8개의 점을 알아야지만 abcdefghi의 값을 알아낼 수 있음

#### 동차 좌표와 동차 행렬

- 동차 좌표

  ![1569992983964](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569992983964.png)

- 동차 행렬

  ![1569993003241](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993003241.png)

- 동차 행렬을 이용한 기하 변환

  ![1569993047462](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993047462.png)

  ![1569993060241](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993060241.png)

- 동차좌표 사용 이유

  - 복합 변환을 이용한 계산 효율

    - 이동 후 회전은 두 번의 행렬 곱셈, 복합 변환을 이용하면 한 번의 곱셈

    ![1569993212813](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993212813.png)

  - 임의의 점 ![1569993226899](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993226899.png)을 중심으로 회전

    ![1569993237130](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993237130.png)

#### 영상에 적용

- 전방 변환은 심한 에일리어싱 현상

  - 계단 현상이 생길 수 있음(구멍이 생긴다)

    - 주변에 있는 값으로 값을 매꿔줌(평균을 내서)

    ![1569993344394](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993344394.png)

- 후방 변환을 이용한 안티 에일리어싱

  ![1569993286369](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993286369.png)
  ![1569993365636](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993365636.png)

- 보간에 의한 안티 에일리어싱

  - 실수 좌표를 반올림하여 정수로 변환하는 과정에서 에일리어싱 발생

  - 주위 화소 값을 이용한 보간으로 안티 이일리어싱![1569993409080](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993409080.png)

  - 양선형 보간

    ![1569993425104](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993425104.png)

    - 거리가 먼 쪽에 더 weight, 가까운 쪽에 덜 weight

    ![1569993476435](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993476435.png)

- 최근접 이웃, 양선형 보간, 양 3차 보간의 비교

  ![1569993530930](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993530930.png)

  - 최근접 이웃으로 할 시, 계단현상 발생(블럭 생김)

## 다해상도

- 해상도를 줄이거나 늘리는 연산

  - 다양한 응용
    - 멀티미디어 장치에 디스플레이
    - 물체 크기 변환에 강인한 인식 등
  - 업샘플링과 다운샘플링
    - 업샘플링 : 작은이미지에서 큰 이미지로 만드는 것(하나의 픽셀 -> 4개의 픽셀)
    - 다운샘플링 : 큰 이미지에서 작은 이미지로 만드는 것(4개의 픽셀 -> 하나의 픽셀)

- 피라미드

  - 샘플링 비율 0.5로 다운샘플링

    ![1569993661214](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993661214.png)

  - 구축 연산

    ![1569993926506](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993926506.png)

    - 에일리어싱 발생(화소에 따라 100% 또는 0%만큼 공헌)

  - Burt & Adelson 방법 [Burt83a]

    ![1569994012478](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569994012478.png)

    - 모든 화소가 50%씩 공헌

      ![1569994033802](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569994033802.png)

  ![1569994095582](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569994095582.png)

  ![1569993969566](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569993969566.png)

## 모폴로지

- 원래 생물학에서 생물의 모양 변화를 표현하는 기법
- 수학적 모폴로지 : 컴퓨터 비전에서 패턴을 원하는 형태로 변환하는 기법

### 이진 모폴로지

- 구조 요소

  ![1569994366044](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569994366044.png)

- 팽창, 침식, 열기, 닫기 연산

  ![1569994399053](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569994399053.png)

  - 팽창 : 내가 1이면 주변 값 다 채움(상하좌우)
  - 침식 :  상하좌우 중 하나라도 없으면 본인이 사라짐(상하좌우 모두 있으면 남겨짐)
  - 열기 : 침식을 먼저하고 그 결과에 팽창(잡음을 제거하기 위함)
  - 닫기 : 팽창을 먼저하고 그 결과에 침식(작은 구멍들을 제거하기 위함)

  ![1569994645060](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569994645060.png)

### 명암 모폴로지

- **<u>잘 쓰지 않음</u>**

![1569994992612](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569994992612.png)

![1569995004361](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569995004361.png)
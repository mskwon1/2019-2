# Filter

## 영상처리의 세 가지 기본 연산

### 점 연산

- 오직 자신의 명암값에 따라 새로운 값을 결정

- 식으로 쓰면

  - 대부분은 k=1(한 장의 영상을 변환)

    ![1569818425073](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569818425073.png)

- 선형 연산

  ![1569818437760](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569818437760.png)

  ![1569818456428](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569818456428.png)

- 비선형 연산

  - 예) 감마 수정(모니터나 프린터 색상 조절에 사용)

    ![1569818505305](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569818505305.png)

- 디졸브

  - k = 2인 경우

    ![1569818533195](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569818533195.png)
    - 알파 값이 1이면 뒷항이 없어지고, 0이면 앞 항이 없어짐
    - 0~1 비율을 조절해 두 개의 영상을 결합

### 영역 연산(Filtering)

- 이웃 화소의 명암값에 따라 새로운 값 결정
- Image filters in spatial domain(영상평면에서의 연산)
  - Filer is a mathematical operation of a grid of numbers
  - smoothing, sharpening, measuring texture
- Image Filtering : compute function of **local neighborhood** at each position
- 매우 중요함
  - Enhance images
    - denoise, resize, increase contrast
  - **Extract information from images**
    - texture, edges, distinctive points
  - Detect patterns
    - template matching

#### 상관과 컨볼루션

- 상관

  - 원시적인 매칭 연산 (물체를 윈도우 형태라고 표현하고 물체를 검출)

    - 윈도우와 가장 유사한 위치 찾기

    - **최대값 29를 갖는 위치6에서 물체 검출**

      ![1569818823175](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569818823175.png)

    - 영상은 고차원 데이터(엄청 큼)

      - 차원을 줄이기 위해 컨볼루션 이용, 의미가 있을 것 같은 정보들을 뽑아서 인식
      - 앞부분에 불필요한 차원을 줄이고 의미있는 정보(내가 원하는 정보)만 추출

- 컨볼루션

  - 윈도우를 **뒤집은 후** 상관 적용
  - 임펄스 반응

- 2차원

  ![1569819057816](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569819057816.png)

  - 상관 대신 컨볼루션을 하면, **윈도우와 똑같은 형태를 가지도록 아웃풋을 만들 수 있다**

- 수식 표현

  ![1569819170171](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569819170171.png)

- 따로 둘을 구분하지 않고 컨볼루션이라는 용어 사용

- 컨볼루션 예제

  - 박스와 가우시안은 스무딩 효과
  - 샤프닝은 명암 대비 강조
  - 수평 에지와 수직 에지는 에지검출 효과

  ![1569819456088](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569819456088.png)

- 컨볼루션은 선형 연산

  - 주요 특성
    - Linearity : filter(f1 + f2) = filter(f1) + filter(f2)
    - Shift Invariance : Same behavior regardless of pixel location
      - filter(shift(f)) = shift(filter(f))
    - 모든 linear, shift invariant 연산자가 컨볼루션이라고 볼 수 있음
  - 다른 특성
    - Commutative : a * b = b \* a
    - Associative : a * (b \* c ) = (a \* b) \* c
    - 분배법칙
    - Scalaras factor out
    - Identity

##### 박스 필터 예제

- 각 픽셀을 주변 픽셀의 평균값으로 대체
- 스무딩 효과

![1569819743276](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569819743276.png)

- 박스 필터의 설정에 따라 이미지 필터링 효과

![1569819908905](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569819908905.png)

![1569819898001](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569819898001.png)

![1569819929655](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569819929655.png)

![1569820084338](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569820084338.png)

- 좌우의 명암차가 큰 영역만 뽑기

![1569820110682](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569820110682.png)

- 상하의 명암차가 큰 영역만 뽑기

##### 중요필터 : 가우시안

- Weight contributions of neighboring pixels by nearness

  ![1569820301603](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569820301603.png)

  - 박스필터보다 스무스하게 느껴짐

- Remove "high-frequency" components from the image(low-pass filter)

  - Images become more **smooth**

- Convolition with self is another gaussian

- Separable Kernel(**Seperability**)

  - The 2D Gaussian can be expressed as the product of two functions, one a function of x and the other a function of y
    - 각각의 Function은 1D Gaussian

  ![1569820467248](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569820467248.png)
  - Example

    ![1569820518277](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569820518277.png)

    - 2차원을 한꺼번에 하는건 복잡하니 1차원씩 나눠서 할 수 있다
    - tensor(3차원) 역시 마찬가지로 쪼갤 수 있다 = tensor decomposition

#### 비선형 연산

- 메디안 필터

  - 솔트페퍼 잡음(하얀색, 검은색 잡음)
  - 가우시안에 비해 에지 보존 효과가 뛰어남

  ![1569820790181](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569820790181.png)

  ![1569820854810](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569820854810.png)

- 스무딩 + 잡음 제거 유용

# 에지 검출

- 에지의 유용성
  - 물체의 경계를 표시해 줌
  - 매칭에 용이한 선분이나 곡선으로 변환 가능
- 에지의 한계
  - 실종된 에지(거짓 부정), 거짓 에지(거짓 긍정) 발생
  - 이들 오류를 어떻게 최소화할 것인가?

## 에지 검출의 기초

- 원리

  - 물체 내부나 배경은 변화가 없거나 작은 반면, 물체 경계는 변화가 큼
  - 이 원리에 따라 에지 검출 알고리즘은 명암, 컬러, 또는 텍스처의 변화량을 측정하고, 변화량이 큰  곳을 에지로 검출

  ![1570957246965](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957246965.png)

### 디지털 영상의 미분

- 1차원

  - 연속 공간에서 미분

    ![1570957489682](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957489682.png)

  - 디지털(이산) 공간에서 미분

    ![1570957503842](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957503842.png)

### 에지 모델과 연산자

- 계단 에지와 램프 에지

  - 자연 영상에서는 주로 램프 에지가 나타남

    ![1570957534063](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957534063.png)

- 2차 미분

  ![1570957552925](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957552925.png)

- 램프 에지에서의 미분의 반응

  ![1570957564231](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957564231.png)

- 에지 검출 과정

  - 1차 미분에서 봉우리 또는 2차 미분에서 영교차를 찾음
  - 두꺼운 에지에서 위치 찾기 적용

- 현실에서는 잡음 때문에 스무딩 필요

  - $\triangle x = 2$인 연산자로 확장

    ![1570958054369](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958054369.png)

  - 2차원으로 확장

    ![1570958063402](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958063402.png)

- 정방형으로 확장하여 스무딩 효과

  ![1570958075436](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958075436.png)

#### Closed Up of Edges

![1570957669513](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957669513.png)

#### Characterizing Edges

- An edge is a place of rapid change in the image intensity function

  ![1570957715987](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957715987.png)

#### Intensity Profile

![1570957734524](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957734524.png)

#### With a little Gaussian noise

![1570957748385](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957748385.png)

#### Effects of Noise

- Conside a single row or column of the image

  - Plotting intensity as a function of position gives a signal

    ![1570957775799](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957775799.png)

- Difference filters respond strongly to noise

  - Image noise results in pixels that look very different from their neighbors
  - Generally, the larger the noise the stronger the response

- What can we do about it?

#### Solution : smooth first

![1570957838471](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957838471.png)

#### Derivative Therorem of convolution

- Differentiation is convolution, and convolution is associative

  ![1570957867051](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957867051.png)

- This saves us one operation

  ![1570957881186](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957881186.png)

#### Derivative of Gaussian filter

![1570957896819](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957896819.png)

- 분리 가능한가?

![1570957922633](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570957922633.png)

- 어느 쪽이 가로/세로 Edge?

### 에지 강도와 에지 방향

- 에지 검출 연산

  ![1570958115074](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958115074.png)

  ![1570958125719](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958125719.png)

  ![1570958135638](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958135638.png)

## 영교차 이론

- 1980년에 Marr와 Hildreth가 개발 [Marr80]
  - 이전에는 주로 소벨을 사용

### 가우시안과 다중 스케일 효과

- 가우시안을 사용하는 이유

  - 미분은 잡음을 증폭시키므로 스무딩 적요이 중요함

    ![1570958181722](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958181722.png)

  - $\sigma$를 조절하여 다중 스케일 효과

  - 에지의 세밀함 조절 가능

    ![1570958236270](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958236270.png)

- 가우시안

  ![1570958264960](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958264960.png)

  - $\sigma$로 스케일 조절

  ![1570958282905](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958282905.png)

- 2차원 가우시안

  ![1570958296981](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958296981.png)

- 이산 공간에서 구현

  - 마스크 크기가 작으면 오차, 크면 계산 시간 과다
  - 6$\sigma$와 같거나 큰 가장 작은 홀수

### LOG필터

- Marr-Hildreth 에지 검출 알고리즘 [Marr80]

  - 2차 미분에서 영교차 검출

  ![1570958348056](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958348056.png)

- 라플라시안 (2행)

  ![1570958359076](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958359076.png)

- LOG 필터

  - 입력 영상에 가우시안 G를 적용한 후, 결과에 라플라시안을 다시 적용하는 두 단계의 비효율성

    - 계산시간 과다
    - 이산화에 따른 오류 누적

  - LOG 필터를 이용한 한 단계 처리

    ![1570958396051](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958396051.png)

    ![1570958408461](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958408461.png)

    ![1570958419577](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958419577.png)

- 영교차 검출 (2행)

  - 여덟 개의 이웃 중에 마주보는 동-서, 남-북, 북동-남서, 북서-남동의 화소 쌍 네 개를 조사한다. 그들 중 두 개 이상이 서로 다른 부호를 가진다
  - 부호가 다른 쌍의 값 차이가 임계값을 넘는다

  ![1570958471658](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958471658.png)

  ![1570958481543](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958481543.png)

## 캐니 에지

- 앞 절은 그럴듯해 보이는 에지 연산자 사용

- 1986년에 Canny 에지 발표 [Canny86]

  - 에지 검출을 최적화 문제로 해결
  - 세 가지 기준
    - 최소 오류율 : 거짓 긍정과 거짓 부정이 최소여야 한다. 즉, 없는 에지가 생성되거나 있는 에지를 못 찾는 경우를 최소로 유지해야 한다
    - 위치 정확도 : 검출된 에지는 실제 에지의 위치와 가급적 가까워야 한다
    - 에지 두께 : 실제 에지에 해당하는 곳에는 한 두께의 에지만 생성해야 한다

- 알고리즘

  ![1570958773637](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958773637.png)

- 비최대 억제

  - 이웃 두 화소보다 에지 강도가 크지 않으면 억제됨

    ![1570958789263](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958789263.png)

- 이력 임계값

  - 두 개의 임계값 $T_{high}$와 $T_{low}$사용하여 거짓 긍정 줄임
  - 에지 추적은 $T_{high}$를 넘는 화소에서 시작, 추적 도중에는 $T_{low}$적용

  ![1570958840125](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958840125.png)

  ![1570958852218](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958852218.png)

![1570958863150](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958863150.png)

### The Canny detector

![1570958567213](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958567213.png)

![1570958580388](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958580388.png)

#### Non-maximum suppression

![1570958608105](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958608105.png)

- Check if pixel is local maximum along gradient direction, select single max across width of the edge
  - requires checking interpolated pixels p and r

![1570958660002](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958660002.png)

#### Hysteresis thresholding

- Use a high threshold to start edge curves, and a low threshold to continue them

  ![1570958696831](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958696831.png)

## 컬러 에지

- RGB 채널에 독립적으로 적용 후 OR 결합

  - 에지 불일치 발생

    ![1570958894767](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958894767.png)

- 디 젠조 방법

  ![1570958906061](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958906061.png)
  ![1570958918168](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570958918168.png)
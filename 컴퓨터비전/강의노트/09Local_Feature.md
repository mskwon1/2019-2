# 지역 특징 검출

- 대응점 찾기
  - 같은 장면을 다른 시점에서 찍은 **두 영상에서 대응하는 점의 쌍**을 찾는 문제
  - 파노라마, 물체 인식/추적, 스테레오 등 컴퓨터 비전의 중요한 문제 해결의 단초
    - 검출 -> 기술(Descriptor) -> 매칭의 세 단계로 해결

## 지역 특징 검출의 기초

### 특징 검출의 역사 : 지역 특징의 대두

- 무엇을 특징점으로 쓸 것인가?
  - 에지 : 에지 강도와 방향 정보만 가지므로, 매칭에 참여하기에 턱없이 부족
- 다른 곳과 두드러지게 달라 풍부한 정보 추출 가능한 곳
  - 에지 토막에서 곡류이 큰 지점을 코너로 검출
    - 코너 검출, dominant point 검출 등의 주제로 80년대 왕성한 연구
    - 90년대 소강 국면, 2000년대 사라짐
    - 더 좋은 대안이 떠올랐기 때문
  - **지역 특징**이라는 새로운 물줄기
    - 명암 영상에서 직접 검출
    - 의식 전환 : 코너의 물리적 의미 :arrow_forward: 반복성

### 지역 특징의 성질

- 지역 특징
  - <위치, 스케일, 방향, 특징 벡터> = ((y,x), s, $\theta$, **x**)로 표현
    - 검출단계 : 위치와 스케일 알아냄
    - 기술단계 : 방향과 특징 벡터 알아냄
- 지역 특징이 만족해야 할 특성
  - 반복성
  - 분별력
  - 지역성
  - 정확성
  - 적당한 양
  - 계산 효율
    - 차원을 줄일 방법(영상은 고차원 데이터)
- 이들 특성은 길항 관계
  - 응용에 따라 적절한 특징을 선택해야 함
    - 정해진 '적당한 양'이라는 것은 없다

### 지역 특징 검출 원리

- 원리
  - 인지 실험
    - 대응점을 찾기가 쉬운 점은? :arrow_forward: 사람에게 쉬운 곳이 컴퓨터에게도 쉽다
  - 좋은 정도를 어떻게 수량화할까?
    - 여러 방향으로 **밝기 변화**가 나타나는 곳일수록 높은 점수
    - 최소 3방향에 차이가 있으면 특징으로 검출

### Corner Detection : Basic Idea

- We should easily recognize the point by looking through a small **window**

- Shifting a window in any direction should give a large change in intensity

  ![1572843810531](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572843810531.png)

## 이동과 회전에 불변한 특징점 검출

- 어떻게 특징을 찾을것인지

### 모라벡 알고리즘

- 인지 실험에 주목한 모라벡 [Moravec80]

  - 제곱차의 합으로 밝기 변화 측정

    ![1572843924821](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572843924821.png)

    ![1572843946226](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572843946226.png)

    - w(y,x)가 확률 변수
    - 이 값이 크다 : 나를 기준으로 주변 값의 변화량이 크다

    ![1572843993860](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572843993860.png)

    - 플랫 : 변화값 거의 없음
    - 에지 / 코너 : 변화값 많음

    ![1572844015565](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572844015565.png)

    - a : 코너, b : 에지, c : 플랫

- 모라벡이ㅡ 함수

  - 특징 가능성 값 $C$

    ![1572845014394](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572845014394.png)

    - **미니멈 값이 1 이상이 되면 '코너'로 칭함**
    - 대각선 방향은 안본다

  - 한계

    - 한 화소만큼 이동하여 **네 방향**만 봄
    - 잡음에 대한 대처 방안 없음

### 해리스 코너

- 해리스의 접근 [Harris88]

  - 가중치 제곱차의 합을 이용한 잡음 대처

    ![1572845186428](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572845186428.png)

  - 테일러 확장 ![1572845215710](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572845215710.png)을 대입하면,

    ![1572845241172](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572845241172.png)

    ![1572845450947](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572845450947.png)

    - $u, v$는 이 수식에 영향을 받지 않는다
    - $d_y^2$ : y방향으로 얼마나 차이나는지, $d_x^2$ : x방향으로 얼마나 차이나는지
    - 모두 0이면 플랫
    - 몇 개의 픽셀을 보던지 간에 u,v는 영향을 받지 않고, 
      x축 방향/y축 방향의 분산값이 코너점을 결정

- 2차 모멘트 행렬 A

  ![1572845804592](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572845804592.png)

  - $(v,u)$는 실수 가능
  - **A**를 $(v,u)$ 무관하게 계산할 수 있음(S가 u와 A의 곱으로 인수분해되어 있으므로)
  - **A**는 **영상 구조**를 나타냄 -> **A**를 잘 분석하면 특징 여부를 판정할 수 있음

![1572846130936](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572846130936.png)

![1572846160881](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572846160881.png)

- 2차 모멘트 행렬의 고유값 분석

  - c와 같이 두개의 고윳값 모두 0이거나 0에 가까우면 -> 변화가 거의 없는 곳

  - b와 같이 고유값 하나는 크고 다른 하나는 작으면 -> 한 방향으로만 변화가 있는 에지

  - a와 같이 고유값 두개가 모두 크면 -> 여러 방향으로 변화가 있는 지점, **특징점으로 적합**

    ![1572846226641](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572846226641.png)

- 특징 가능성 값 측정

  ![1572846255588](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572846255588.png)

  - 고유값 계산을 피해 속도 향상

    ![1572846270608](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572846270608.png)

![1572846289233](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572846289233.png)

- 위치 찾기 문제 대두
  - 큰 C값을 가진 큰 점들이 밀집되어 나타나므로 대표점 선택 필요
- 코너라는 용어가 적절한가?
  - 코너 :arrow_forward: 특징점 또는 관심점

#### Corner Detection : Mathematics

![1572845975335](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572845975335.png)

![1572846025508](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572846025508.png)

##### Interpreting the eigen values

![1572846089696](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572846089696.png)

![1572846105060](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572846105060.png)

### 2차 미분을 사용한 방법

### 슈산
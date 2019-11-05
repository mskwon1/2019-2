## 선분 검출

- 에지 맵 -> 에지 토막 -> 선분

### 에지 연결과 선분 근사

![1571202763240](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571202763240.png)

- 체인 코드 : 시작하는점 ~ 방향숫자

- 선분 근사

  - 두 끝점을 잇는 직선으로부터 가장 먼 점 까지의 거리 h가 임계값 이내가 될 때까지 선분 분할을 **재귀적으로 반복**

    ![1571202930501](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571202930501.png)

  - 차원을 줄이겠다는 의도

### 허프 변환(Hough Transform)

- 에지 연결 과정 없이 선분 검출 (전역 연산을 이용한 지각 군집화)

- 영상 공간 y-x를 기울기 절편 공간 b-a로 매핑

  ![1571203032489](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571203032489.png)

  - 차원을 줄이면서도 특성을 유지한다

- 수직선의 기울기가 $\infin$문제

  - 극 좌표계 사용하여 해결

    ![1571203207385](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571203207385.png)

  - p에 따라서 $\theta$값이 어떻게 변화하는가

- 밀집된 곳 찾기

  - 양자화된 누적 배열 이용하여 해결

    ![1571203677894](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571203677894.png)

- 원 검출

  - 3차원 누적 배열 사용

    ![1571203691332](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571203691332.png)

  ![1571203706562](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571203706562.png)

- An early type of voting scheme

#### Voting Schemes

- Let each feature vote for all the models that are compatible with it
- Hopefully the noise features will not vote consistently for any single model
- Missing data doesn't matter as long as there are enough features remaining to agree on a good model

![1571203800058](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571203800058.png)

- 주어진 에지 영역이 어떤 특징을 가지고 있는가(몇 개의 점들만 가지고 직선방정식을 그린다)
  - $\theta$와 p의 극좌표계 이용

#### General Outline

- Discretize parameter space into bins

- For each feature point in the image, put a vote in every bin in the parameter space that could have generated this point

- Find bins that have the most votes

  ![1571203996860](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571203996860.png)

#### Parameter Space Representation

- A line in the image corresponds to **a point** in Hough Space

  ![1571204041162](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571204041162.png)

- What does a point ($x_0, y_0$) in the image space map to in the Hough space

  ![1571204099971](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571204099971.png)

  ![1571204127680](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571204127680.png)

  - the solutions of $b = -x_0m + y_0$
  - this is **a line** in hough space

- Where is the line that contains both  ($x_0, y_0$) and  ($x_1, y_1$)

  ![1571204196446](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571204196446.png)

  - the intersection of $b = -x_0m + y_0$ and  $b = -x_1m + y_1$

- Problems with the (m,b) space

  - Unbounded parameter domains
  - Vertical lines require infinite m

- Alternative : polar representation

  ![1571204257020](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571204257020.png)

  - each point will add a sinusoid in the ($\theta,p$) parameter space

#### Algorithm Outline

- Initialize accumulator H to all zeros

- For each edge point (x,y) in the image

  ![1571204305487](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571204305487.png)

- Find the values of ($\theta,p$) where H ($\theta,p$) is a local maximum

  - the detected line in the image is given by

    $p = x cos\theta + y sin\theta$

#### Basic Illustration

![1571204378573](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571204378573.png)

#### Other shapes

![1571204399480](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571204399480.png)

#### Sevaral Lines

![1571204418217](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571204418217.png)

#### A more complicated image

![1571204433260](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571204433260.png)
- 많이 겹치는 곳 : 세타가 똑같은 친구들이 겹쳐있는 것
- 허프 트랜스폼을 잘 쓰면 다양한 잡음을 제거할 수 있다

#### Effect of Noise

- 멀리 본다 : 노이즈 증가
- 가까이 본다 : 정교한 제어 필요

#### Random Points

- Uniform  noise can lead to spurious peaks in the array

  ![1572237962748](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572237962748.png)

- As the level of noise increases, the maximum number of votes increases too

#### Dealing with noise

- Choose a grid / discretization
  - Too coarse : large  votes obtained when too many different lines correspond to a single bucket
  - Too fine : miss lines because some points that are not exactly collinear cast votes for different buckets
- Increment neighboring bins (smoothing in accumulator array)
- Try to get rid of irrelevant features
  - Take only edge points with significant gradient magnitude

#### Incorporating image gradients

![1572238253283](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238253283.png)

![1572238264778](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238264778.png)

#### Hough tranform for circles

![1572238348652](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238348652.png)

- How many dimensions will the parameter space have?

- Given an oriented edge point, what are all possible bins that it can vote for?

- 반드시 반지름 $r$ 값의 범위가 주어져야 함

- Conceptually equivalent procedure : for each (x,y,r), draw the corresponding circle in the image and compute its "support"

  ![1572238592709](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238592709.png)

#### Generalized Hough transform

- We want to find a template defined by its reference point(center) and sveral distict types of landmark points in stable spatial configuration

  ![1572238635694](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238635694.png)

- Template representation : for each type of landmark point, store all possible displacement vectors towards the center

  ![1572238681947](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238681947.png)

- Detecting the template : for each feature in a new image, look up that feature type in the model and vote for the possible center locations associated with that type in the model

  ![1572238746578](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238746578.png)

#### Application in recognition

- Index displacements by 'visual codeword'

  ![1572238790207](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238790207.png)

  - 바퀴라는 특징을 가지는 모델

  ![1572238831363](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238831363.png)
  
  - 두 개 이상의 바퀴가 있는 곳에 자동차가 있을 것이라는 것을 추측가능

##### Implicit shape models : Training

- Build codebook of patches around extracted interest points using clustering

  ![1572238892375](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238892375.png)

- Map the patch around each interest point to closet codebook entry

  ![1572238914494](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238914494.png)

- For each codebook entry, store all positions it was found, relative to object center

  ![1572238941513](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238941513.png)

- Extract weighted segmentation mask based on stored masks for the codebook occurrences

  ![1572238975975](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572238975975.png)

  - 자동차의 중심을 찾아냄

##### Implicit Shape Models : Details

- **Supervised** Training
  - need reference location and segmentation mask for each training car
- Voting space is continuous, not discrete
  - Clustering algorithm needed to find maxima
- How about dealing with scale changes?
  - Option 1 : Search a range of scales, as in Hough transform for circles
  - Option 2 : Use interest points with characteristic scale
- Verification stage is very important
  - Once we have a location hypothesis, we can overlay a more detailed template over the image and compare pixel-by-pixel, transfer segmentation masks, etc

#### Hough Transform : Discussion

- Pros
  - Can deal with non-locality and occlusion
  - Can detect multiple instances of a model
  - Some **robustness to noise** : noise points unlikely to contribute consistently to any single bin
- Cons
  - Complexity of search time increases exponentially with the number of model parameters
  - Non-target shapes can produce spurious peaks in parameter space
  - It's har to pick a good grid size

### RANSAC

- 1981년 Fischler & Bolles가 제안 [Fischler81]

- 인라이어를 찾아 어떤 모델을 적합시키는 기법

- 난수 생성하여 인라이어 군집을 찾기 때문에 임의성을 지님

- 원리

  - 선분 검출에 적용

  - 모델은 직선의 방정식 $y = ax + b $

    ![1572239880714](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572239880714.png)

- 매칭 쌍 집합 $X = \{(a_1,b_1),(a_2,b_2),...,(a_n,b_n)\}$을 처리할 수 있게 확장![1572240226437](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572240226437.png)

  ![1572240301503](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572240301503.png)

  ![1572240321015](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572240321015.png)

  


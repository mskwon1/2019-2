## 스케일에 불변한 특징점 검출

- 거리에 따른 스케일 변화

  - 멀면 작고 윤곽만 어렴풋이 보이다가, 가까워지면 커지면서 세세한 부분 보임

  - 사람은 강인하게 대처하는데, 컴퓨터 비전도 대처 가능한가

    <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191111142138366.png" alt="image-20191111142138366" style="zoom: 67%;" />

### Types of Invariance

- Illumination

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191111142256373.png" alt="image-20191111142256373" style="zoom:67%;" />

- Scale

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191111142308646.png" alt="image-20191111142308646" style="zoom:67%;" />

- Rotation

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191111142322222.png" alt="image-20191111142322222" style="zoom:67%;" />

- Affine

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191111142335149.png" alt="image-20191111142335149" style="zoom:67%;" />

### SIFT(Scale Invariant Feature Transform)

- An algorithm in computer vision to detect and describe local features in image. The features are invariant to image scaling, rotation, affine, occlusion and illumination. Application include object recognition, image stitching and matching, and video tracking, and so on

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191111143058026.png" alt="image-20191111143058026" style="zoom:67%;" />

#### SIFT MATCHING : APPLICATION

- 파노라마 영상 만들기
  
  - image matching 필요
  
- 특징점 매칭
  
  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113133856296.png" alt="image-20191113133856296" style="zoom:67%;" />
  
  - 각각의 영상으로부터 특징점 검출
  - 영상들 간 corresponding pairs를 찾음
  - 영상들을 align 시킴
  
- Problem 1

  - 각 영상으로부터 똑같은 특징점을 찾아야 함
  - need a repeatable detector

- Problem 2

  - 어떻게 똑같은 특징점들을 비교하여 매칭할 것인가
  - need a reliable and distinctive descriptor

#### SIFT : ALGORITHM

1. Find interest points or 'keypoints'
2. Find their dominant orientation
3. Compute their descriptor
4. Match them on other images

### 스케일 공간

- 다중 스케일 영상을 구현하는 두 가지 방식

  - 가우시안 스무딩 : 스케일에 해당하는 $\sigma$가 연속 공간에 정의

    - 주변의 명암값과 확실한 차이가 있는 곳이 keypoint

    ![image-20191113134244821](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113134244821.png)

  - 피라미드 : $\frac{1}{2}$씩 줄어드므로 이산적인 단점

    - 처리시간이 많이 걸린다

    ![image-20191113134302227](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113134302227.png)

- 가우시안 스무딩에 의한 스케일 공간

  - 스케일 축을 추가한 3차원 공간
  - 스무딩을 심하게 해도 잘 남아있는 특징을 찾는다

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113134354095.png" alt="image-20191113134354095" style="zoom:67%;" />

#### SIFT : algorithm

- keypoints are taken as maxima/minima of a **DoG pyramid**

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113135054009.png" alt="image-20191113135054009" style="zoom:67%;" />

  - 옥타브 : 영상 크기가 달라짐
  - 가우시안 영상 간 명암차이가 있는 영역들을 뽑아보겠다(Difference of Gaussian)

- Scale selection(T. Lindeberg)

  - In the absence of other evidence, assume that a scale level, at which some (possibly non-linear) combination of normalized derivatives assumes a local maximum over scales, can be treated as reflecting a characteristics length of a corresponding structure in the data

  ![image-20191113135918457](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113135918457.png)

- Difference of Gaussian

  - Approximation of Laplcian of Gaussian

    ![image-20191113140031051](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113140031051.png)

- DoG pyramid is simple to compute

  ![image-20191113140306799](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113140306799.png)

- Scale space images

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113140403084.png" alt="image-20191113140403084" style="zoom: 50%;" />

- DoG images

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113140436870.png" alt="image-20191113140436870" style="zoom:50%;" />

  - 밝게 나오는 값 : 위에서 아래로 뺐을때 +값이 나온 것
  - 어둡게 나오는 값 : 위에서 아래로 뺐을때 - 값이 나온 것

- Finding extrema

  - sample point is selected only if it is a minimum or a maximum of these points

    <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113140641770.png" alt="image-20191113140641770" style="zoom: 67%;" />

  - then we just find **neighborhood extrema** in this **3D DoG Space**

    <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113140731595.png" alt="image-20191113140731595" style="zoom:67%;" />

    - If a pixel is an extrema in its neighboring region he becomes a **candidate keypoint**
    - 스케일의 변화가 있더라도 일정하게 똑같이 결과가 나오는 것들

- Too many keypoints

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113141543419.png" alt="image-20191113141543419" style="zoom:67%;" />

  - remove low contrast

  - remove edges

    <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191113141633664.png" alt="image-20191113141633664" style="zoom: 50%;" />

  - 피라미드를 옥타브에 따라 만들고, 옥타브마다 sigma값을 조정한 영상들을 만들어 DoG를 구한다

- Each selected keypoint is assigned to oone or more dominant orientations. This step is important to achieve rotation invariance
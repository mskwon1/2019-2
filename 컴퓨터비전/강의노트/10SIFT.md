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
  - 각각의 영상으로부터 특징점 검출
  - 영상들 간 corresponding pairs를 찾음
  - 영상들을 align 시킴


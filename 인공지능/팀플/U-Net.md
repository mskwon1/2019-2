## U-Net

Network 형태가 알파벳 U와 형태가 비슷하게 생겨서 **U-Net**

#### Terms

- Patch
  - 이미지 인식 단위
  - 이미지를 잘라 인식하는 단위(Sliding Window 처럼)

- Contracting Path
  - 이미지를 점점 **줄여 나가는** 부분
- Expanding Path
  - 이미지를 **키워 나가는** 부분
- Context
  - 이웃한 픽셀들 관의 관계
  - 이미지의 '문맥'

### 특징

Reference : 논문 「U-Net : Convolutional Networks for Biomedical Image Segmentation」

- Classification + Localization의 역할

#### 기존 Segmentation Network들의 문제점 해결

- 느린 속도

  - Overlap 비율의 문제

    - Sliding Window 방식을 사용 할 시, 이미 사용한 Patch 구역을 다음 Sliding Window에서 다시 검증
      :arrow_forward: 검증이 끝난 부분을 다시 검증하는 셈, 똑같은 일 반복하는 비효율
    - U-net은 이미 검증이 끝난 부분은 아예 건너뛰고 다음 Patch 부분부터 검증, 따라서 속도가 빠르다

    ![image-20191218165225955](C:\Users\mskwon\Desktop\2019-1\typora_images\image-20191218165225955.png)

  - Fully Connected Layer

    - U-net은 Fully-Connected Layer가 없다

- Trade-Off의 늪
  - Patch-Size의 문제
    - Patch-Size가 커질 시, 더 넓은 범위의 이미지를 한 번에 인식 
      :arrow_forward: Context 인식은 유리, Localization은 불리
    - Patch-Size가 작을 시, 좁은 범위의 이미지만 인식
      :arrow_forward: Localization은 유리, Context 인식은 불리
  - U-net은 여러 Layer의 Output을 동시에 검증해 Localizaion과 Context 인식 모두 원활하게 가능

#### Image Mirroring

- Contracting Path에서 padding이 없기 때문에, 점점 이미지의 외곽 부분이 잘려 ouput의 사이즈가 작아진다
  - 이를 해결하기 위해 Image Mirroring 선택
- 없어지는 부분을 zero-padding 하지 않고, mirror padding
  - 안쪽 이미지를 거울에 반사시킨 형태

#### Network Architecture

- 컨볼루션 층

  - 3x3 Convolution 사용
  - 활성함수 : ReLU

- 풀링 층

  - 2x2 MAX Pooling
    - 매 층마다 $\frac{1}{2}$ 다운샘플링, 채널은 2배

- Expanding Path

  - 2x2 UP Convolution
  - 같은 계층 안에선 3x3 Convolution

- Mirror Padding을 진행할 때 손실되는 Path에 대한 보상

  - Contracting Path의 데이터를 적당한 크기로 crop한 후 concat하는 방식으로 보상(이미지의 회색 선)

    <img src="https://mblogthumb-phinf.pstatic.net/MjAxODA4MDZfMjkg/MDAxNTMzNTUxOTUxOTU0.YzYd-ho-1jFLlBmDWRTlnxRjjKlA2XX0wmutkUXARrcg.r_RiV19V9ocbF_9jM_D9kze0TdFf5oWKY7rnZHQYLIUg.PNG.worb1605/image.png?type=w800" alt="img" style="zoom:67%;" />
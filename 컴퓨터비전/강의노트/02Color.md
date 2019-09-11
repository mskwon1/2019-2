# Color

- Result of interaction between physical light in the environment and our visual system
- Psychological property of our visual experiences when we look at objects and lights, not a physical property of those objects or lights

- Checker Shadow Illusion

  ![1568093386045](../../typora_images/1568093386045.png)

  - Possible Explanations
    - Simultaneous contrast
    - Reflectance edges **VS** Illumination edges

- There is no simple functional description for 
  the perceived color of all lights under all viewing conditions

  - Helpful Constraint : Consider only physical spectra with normal distributions

    ![1568093610385](../../typora_images/1568093610385.png)

    ![1568093621088](../../typora_images/1568093621088.png)

    ![1568093632780](../../typora_images/1568093632780.png)

    ![1568093641974](../../typora_images/1568093641974.png)

- The Physics of Light

  - Example of the reflectance spectra of surfaces

    ![1568093681577](../../typora_images/1568093681577.png)

## The Eye

![1568093483309](../../typora_images/1568093483309.png)

- Ratio of **L** to **M** to **S** cones : Approx. **10:5:1**
- Almost no S cones in the center of the fovea

## Linear Color Spaces

- How to compute the weights of the primaries to match any spectral signal?

  - Given : A choice of **three primaries** and a **target color signal**

  - Find : **Weights of the primaries** needed to match the color signal

    ![1568093788607](../../typora_images/1568093788607.png)

- Also need to specify **matching functions**

  - The amount of each primary needed to 
    match a monochromatic(단색의) light source at each wavelength

    ![1568093870011](../../typora_images/1568093870011.png)

- RGB 모델

  - 길이가 1인 정육면체로 색을 표현

    ![1568093915269](../../typora_images/1568093915269.png)

  - 영상 표현

    ![1568093944042](../../typora_images/1568093944042.png)

- HSI 모델

  - 이중 콘으로 색을 표현

    ![1568093961930](../../typora_images/1568093961930.png)

## Nonlinear Color Spaces : HSV

- Perceptually meaningful dimensions : **H**ue, **S**aturation, **V**alue(Intensity)

- RGB cube on its vertex(꼭지점)

  ![1568094040086](../../typora_images/1568094040086.png)

## 컬러 영상 처리

- 가장 단순한 방법 : 세 채널을 독립적으로 처리

  ![1568094112412](../../typora_images/1568094112412.png)

  ![1568094121726](../../typora_images/1568094121726.png)

## Uses of Color in CV

- Color histograms for **image matching**

  ![1568094165103](../../typora_images/1568094165103.png)

- Multicolr

  ![1568094215523](../../typora_images/1568094215523.png)

- Image Segmentation

  ![1568094237170](../../typora_images/1568094237170.png)

- Skin Detection

  ![1568094256604](../../typora_images/1568094256604.png)

- Robot Soccer

  ![1568094281666](../../typora_images/1568094281666.png)

- 　

  ![1568094332904](../../typora_images/1568094332904.png)
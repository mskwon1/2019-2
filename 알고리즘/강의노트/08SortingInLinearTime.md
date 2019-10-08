# Sorting in Linear Time

- 모든 비교 정렬 알고리즘은 최악의 경우 ![1569853473285](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853473285.png)번의 비교 필요
- 정렬 알고리즘의 실행은 결정 트리의 루트에서 하나의 리프까지 경로를 따라가는 것 

![1569853536000](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853536000.png)

## Counting Sort

- 각 원소는 [0, k] 인 정수인 경우

- 각 원소에 대해 그보다 작은 원소의 갯수를 세면 정렬 후 원소의 위치를 알 수 있다

  ![1569853574808](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853574808.png)

  ![1569853614066](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853614066.png)

  ![1569853628202](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853628202.png)

- Stable sort

  - 출력 배열에서 값이 같은 숫자가 입력 배열에 있던 것과 같은 순서를 유지하는 정렬

  - running time : ![1569853693788](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853693788.png)

    ![1569853715726](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853715726.png)

## Radix Sort

- **가장 낮은 자리 숫자부터** 정렬하는 것을 자리수만큼 반복

  ![1569853786282](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853786282.png)

  - 각각의 정렬은 stable 해야 함, counting sort 사용
  - running time : ![1569853810574](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853810574.png)

- 높은 자리부터 정렬하면

  ![1569853825288](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853825288.png)

## Bucket Sort

- 입력이 uniformly distributed in (0,1)인 경우

  ![1569853916069](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853916069.png)

- Time Complexity

  ![1569853967490](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853967490.png)

  ![1569853984291](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569853984291.png)

- Worst-case running time of bucket sort

  ![1569854024140](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569854024140.png)

- Running times of sorting algorithms

  ![1569854045093](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569854045093.png)

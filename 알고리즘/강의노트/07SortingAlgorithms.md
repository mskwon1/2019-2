# Sorting Algorithms

- Running times of sorting algorithms

  ![1569482221541](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482221541.png)

- Comparison Sorts

  - Insertion Sort
    - In-place sorting
    - worst-case running time : ![1569482255332](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482255332.png)
  - Merge Sort
    - Out-of-place sorting
    - running time : ![1569482276914](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482276914.png)
  - Heap Sort
    - In-place sorting
    - running time : ![1569482289657](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482289657.png)
  - Quick Sort
    - In-place sorting
    - worst-case running time : ![1569482317968](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482317968.png)
    - expected running time : ![1569482330761](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482330761.png)

## Heap Sort

### Heap

- (Binary) Heap : heap property, such as A[Parent(i)] >= A[i], 
  									를 만족하는 완전 이진 트리를 배열에 순서대로 저장한 것

- Length of a heap : 배열에 저장된 **모든 원소의 개수**
- Size of a heap : 배열에 저장된 원소 중 **Heap에 속하는 원소의 갯수**

![1569482493000](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482493000.png)

- Examples

  ![1569482522121](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482522121.png)

### Max Heap (최대 힙)

- Heap Property : A[Parent(i)] >= A[i]
- Height of a node : 노드에서 리프에 이르는 **하향 경로 중 가장 긴 것의 edge 개수**
- Height of a heap : height of a root node = ![1569482584980](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482584980.png), where n = size of heap

![1569482605288](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482605288.png)

### Min Heap (최소 힙)

- Heap Property : A[Parent(i)] <= A[i]

### Max Heapify

![1569482675734](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482675734.png)

- Running Time : ![1569482702264](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482702264.png)

### Build-Max-Heap

![1569482729683](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482729683.png)

- Running Time

  - Heap Size가 n인 heap에서 height가 h인 노드들의 갯수 ![1569482771189](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482771189.png)
  - Height가 h인 실행시간 : O(h)

  ![1569482839918](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482839918.png)

### HEAPSORT

![1569482884210](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482884210.png)

![1569482913059](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482913059.png)

- Running Time

  ![1569482949427](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482949427.png)

### Factorials

![1569482973140](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569482973140.png)

## Quick Sort

### Divide and Conquer

- Divide : 배열 A[p..r]을 두 개의 부분 배열 A[p..q-1]과 A[q+1..r]로 분할
- Conquer : 두 개의 부분 배열을 Quicksort
- Combine : 없음

![1569899624406](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569899624406.png)

### Recursive Implementation

![1569899646915](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569899646915.png)

​	![1569899690580](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569899690580.png)

- Running Time

  - Partition이 Balance 되거나 Unbalance 되거나에 따라서 달라짐

    - Balanced : merge sort와 비슷한 속도가 나옴
    - Unbalanced : insertion sort와 비슷한 속도가 나옴

  - Worst Case Partitioning

    - Partition이 항상 0개 원소의 subarray와 n-1개 원소의 subarray로 나눠질 때

    - input이 sort되어 있을 때

      ![1569899860177](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569899860177.png)

  - Best Case Partitioning

    - Partition이 항상 n/2개의 원소를 가진 2 개의 subarray로 나눠질 때

      ![1569899878112](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569899878112.png)

  - Average Case

    - best case 수행시간에 가까움

    - partition이 항상 9:1로 나눠진다면

      ![1569899924908](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569899924908.png)

  ### 랜덤버전 퀵소트

  ![1569899966479](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569899966479.png)
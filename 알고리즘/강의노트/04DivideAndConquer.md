# Divide and Conquer

- Steps

  ![1568795219877](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795219877.png)

## Maximum-subarray Problem

- Price of stock in 17 day-period

  ![1568795245136](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795245136.png)

- 최대 이익은 최소값이나 최대값과 상관이 없다

  ![1568795262686](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795262686.png)

- Brute Force Solution : ![1568795308052](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795308052.png)

  ~~~pseudocode
  for buy_date = 0 to 15
  	for sell_date = buy_date+1 to 16
  		find maximum price[sell_date] - price[buy_date]
  ~~~

### Change in stock prices

![1568795332038](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795332038.png)

- Definition

  - **Input** : An array **A[1..n]** of numbers (음수가 존재한다고 가정)
  - **Output** : Indices **i and j** such that A[i..j] has the greatest sum of any nonempty, contiguous subarray of A, along with the **sum** of the values in A[i..j]

- Brute Force Solution : ![1568795481553](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795481553.png)

  ~~~pseudocode
  max = A[1]; max_i = 1; max_j = 1
  for i = 1 to 16
  	profit = 0
  	for j = i to 16
  		profit = profit + A[j]
  		if (profit > max)
  			max = profit; max_i = i, max_j = j
  ~~~

- Divide and Conquer Solution

  - **Divide** : MaxSubarray(A[low..high])를 low~mid, mid+1~high 두 subarray로 나눈다
  - **Conquer** : 두 subarray를 재귀적으로 푼다
  - **Combine** : 세 subarray 중 최대값을 고른다

![1568795559425](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795559425.png)

![1568795577828](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795577828.png)

- Analysis

  - assume **n** is a power of two

  - base case : ![1568795617148](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795617148.png)

  - recursive case

    ![1568795643607](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795643607.png)

## Matrix Multiplication

- Definition

  ![1568795705178](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795705178.png)

- Brute Force Solution

  ![1568795719762](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795719762.png)

- Divide and Conquer Solution

  - **Divide** : A,B,C 행렬을 1/4씩 나눈다

    ![1568795767451](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795767451.png)

  - **Conquer** 

    ![1568795822120](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795822120.png)

  - **Combine**

    ![1568795835422](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795835422.png)

  ![1568795849673](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795849673.png)

  - Analysis

    - assume n is a power of two

    - base case : ![1568795894147](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795894147.png)

    - recursive case

      ![1568795925618](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568795925618.png)

    

- Strassen's matrix multiplication

  - **Divide** : A,B,C 행렬을 1/4씩 나눈다

    ![1568796011438](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568796011438.png)

  - **Conquer**

    ![1568796037076](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568796037076.png)

  - **Combine**

    ![1568796049791](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568796049791.png)

  ![1568796076961](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568796076961.png)

  ![1568796088056](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568796088056.png)

  ![1568796103964](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568796103964.png)

  - Analysis

    - Assume n is power of two

      ![1568796123892](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568796123892.png)
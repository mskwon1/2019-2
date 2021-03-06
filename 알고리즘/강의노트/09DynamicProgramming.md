# Dynamic Programming

- 표를 만들어 채워가면서 답을 구하는 방법
- Divide and Conquer 와의 차이점 : overlaps in subproblems
- Meaning of "programming" here : tabular method 
- Used in solving optimization problem
  - find an optimal solution, as opposed to the optimal solution

- 순서
  - 최적해의 구조적 특징을 찾는다
  - 최적해의 값을 재귀적으로 정의한다
  - 최적해의 값을 일반적으로 상향식 방법으로 계산한다
  - 계산된 정보들로부터 최적해를 구성한다

## Rod Cutting

- n인치 막대를 잘라서 판매하여 얻을 수 있는 최대 수익 r_n을 찾아라

- 막대를 자르는 비용은 0

- price table

  ![1570514981002](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570514981002.png)

- 4인치 로드를 자르는 법

  ![1570515000127](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570515000127.png)

- 자르는 방법은 2^n가지(각 1칸 별로 자르거나 말거나 경우의 수)

- r_i for i < n 으로부터 r_n을 구할 수 있음

  - Optimal Substructure를 가졌다

  ![1570515070353](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570515070353.png)

- Recursive Top-down implementation

  ![1570515096219](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570515096219.png)

- top down

  ![1570515122778](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570515122778.png)

- bottom up

  ![1570515140924](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570515140924.png)

- Reconstructing a solution : $\theta(n^2)$

  ![1571282085799](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571282085799.png)
  
  

## Matrix Multiplication

- 여러 개의 행렬을 곱할 때 곱셈 순서에 따라 연산 갯수가 달라진다

  ![1570515191562](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570515191562.png)

- 행렬 곱셈의 순서를 정하는 문제(곱셈 연산 X)

  ![1571282127062](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571282127062.png)

- 순서대로

  - 최적해의 구조적 특징을 찾는다

  - 최적해의 값을 재귀적으로 정의한다

    ![1571282166771](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571282166771.png)

  - 최적해의 값을 일반적으로 상향식 방법으로 계산한다

    ![1570515272972](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570515272972.png)

    ![1571282196553](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571282196553.png)

    ![1571282221483](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571282221483.png)

    ![1571282242676](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571282242676.png)
  
    ![1571282310490](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571282310490.png)
  
  - 계산된 정보들로부터 최적해를 구성한다
  
    ![1570515314819](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570515314819.png)
  
- Elements of dynamic programming

  - Optimal Structure : 문제의 최적해가 subproblem의 최적해를 포함한다
  - Overlapping Subproblems : 최적화 문제의 부분 문제를 풀기 위한 재귀 알고리즘이 같은 문제를 반복해서 푼다

## Longest Common Subsequence(LCS)

- Subsequence Z = {z1 ~ zk} of sequence X = {x1 ~ xm} 단조 증가하는 X의 인덱스 시퀀스 {i1 ~ ik} such that xij zj가 있다

  ![1570515440220](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570515440220.png)

- Brute Force approach

  - LCS : Longest Common Subsequence
  - X의 모든 subsequence $X'$를 찾음($2^m$개, m = X의 길이)
  - $X'$이 Y의 subsequence 인지 확인하고 가장 긴 것을 찾음
  - $O(n2^m) n$ = Y의 길이

- Common subsequence Z of X and Y : Z is subsequence of X, and of Y![1570515456093](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570515456093.png)

  ![1571283573144](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571283573144.png)

  ![1571283582498](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571283582498.png)

  ![1571283598730](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571283598730.png)

  ![1571283617298](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571283617298.png)

  ![1571283638864](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571283638864.png)

  ![1571283711287](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571283711287.png)

  - Constructing LCS (STEP 4)

  ![1571283745450](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571283745450.png)

- Maximum subarray나 matrix multiplication을 dynamic programming으로 풀 수 있는가?
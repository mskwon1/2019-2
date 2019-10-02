### Example : The Hiring Problem

- 회사에서 직원을 고용하는데, 직업 소개소에서 매일 한 명씩 지원자를 보냄
- 회사는 지원자를 면접하고 그 지원자가 **첫번째 지원자거나**, **현재 고용되어 있는 직원보다 유능**하면 반드시 고용한다. 이 때 현재 고용되어 있는 직원이 있으면 해고한다
- 면접비용은 1인당 C_i
- 고용비용은 C_h (해고비용 포함)
- C_h > C_i

- 고용 알고리즘

  ![1569389405370](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569389405370.png)

  - Worst Case : ![1569389426683](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569389426683.png)
    - 매번 다음 지원자가 이전 지원자보다 유능

## 확률적 분석

- 가능한 모든 입력 순서에 대한 평균값을 계산
  - 수행시간이라면, Average-Case Running Time 계산
  - 지표확률변수를 사용해 HIRE-ASSISTANT(n)의 비용을 계산

### 지표 확률 변수 : Indicator Random Variable

![1569389502578](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569389502578.png)

- 확률로부터 확률 변수의 기댓값을 쉽게 계산 가능

- 보조정리 5.1

  ![1569389525402](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569389525402.png)

#### 사용 예 1

- 동전을 한 번 던져서 앞면이 나오는 횟수의 기댓값

  ![1569389557790](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569389557790.png)

#### 사용 예 2

- 동전을 n 번 던져서 앞면이 나오는 횟수 X의 기댓값

  ![1569389583763](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569389583763.png)

### 확률적 분석

- 지표 확률 변수를 사용하여 
  HIRE-ASSISTANT(n)의 비용![1569389679911](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569389679911.png) 을 계산해보면(입력의 순서는 random 분포)

  ![1569389655954](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569389655954.png)

  - HIRE-ASSISTANT(n)의 비용(Average Case) ![1569389714403](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569389714403.png)
  - 입력의 분포에 따라 계산하여 **Average-Case Cost**를 계산
    - 각각의 입력에 대한 Cost가 정해져 있음
      :arrow_forward: Average-Case Cost란 이들의 평균을 구하는 것

## 랜덤화된 알고리즘

![1569389763799](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1569389763799.png)

- 입력의 분포와 상관없이 **Expected Cost**를 계산
  - 위 케이스의 결과가 같은 것은 확률적 분석 과정에서 입력 분포를 랜덤으로 가정했기 때문
  - 입력을 알아도 Cost가 정해져있지 않음 :arrow_forward: Expected Cost만 구할 수 있음
# Recurrence(점화식)

- 더 작은 입력에 대한 자신의 값으로 함수를 나타내는 방정식 혹은 부등식

## 치환법(Substitution Method)

- 해의 모양을 추측한다
- 상수들의 값을 찾아내기 위해 수학적 귀납법을 사용하고 그 해가 제대로 동작함을 보인다

![1568862635225](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568862635225.png)

## 재귀 트리 방법(Recursion tree Method)

- 치환법의 1번 단계의 추측을 하기 위해 재귀 트리를 그린다
  - 재귀 트리의 각 노드는 재귀 호출되는 하위 문제 하나의 비용을 나타냄
- 상수들의 값을 찾아내기 위해 수학적 귀납법을 사용하고 그 해가 제대로 동작함을 보인다

![1568862719504](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568862719504.png)

![1568862744387](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568862744387.png)

![1568862759847](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568862759847.png)

![1568862773706](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568862773706.png)

- 수학적 귀납법을 통한 확인

  ![1568862795546](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568862795546.png)

## 마스터 방법(Master Method)

![1568862827901](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568862827901.png)

- Examples

![1568862842304](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1568862842304.png)
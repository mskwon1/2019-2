# PID Control Basics

## Closed Loop Control System

![1570442179975](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570442179975.png)

- Target system examples

  ![1570442209948](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570442209948.png)

## PID Control

- 목표
  - 붕뜨고 반응이 느린 타깃 시스템에 대해 안정적이고 빠른 제어 방법론
- Means
  - P : Proportional gain, present(현재의 상태)
  - I : Integral gain, past(누적, accumulation)
  - D : Derivative gain, future, future( 바로 직전과 지금의 차이)

## Anatomy of Equation

![1570443655705](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570443655705.png)

- Pros
  - Applicable to **many systems without knowing their accurate model**
- Cons
  - Does not guarantee the **optimal control of the target system**

## Parameter Tuning

- Hard to find a systematic way to determine optimal values
- Trial & error / empirical methods
- Known / applicable heuristics
  - Ziegler-Nichols method
  - Twiddle algorithm
- Commercial tools
  - CEMTool / MATLAB & Simulink

## Integrator Windup

- Actuators have **limitations**
  - Saturation : the control variable reaches the **actuator limit**
  - Accumulated error can be **too large** when saturation continues
- Windup may cause oscillation
  - The integral term still remains high when the target system reaches to the desired state
- Solutions
  - Initialize the integral term to a predefined one, or reset to zero
  - Moderately increase the desired state value
  - Disable the integral term for a while
  - Limit the min-max range of the integral term
  - Back-calculate the integral term such that the control variable does not exceed a certain bound
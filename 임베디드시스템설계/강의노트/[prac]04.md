## Light Sensor

### Pin Description

- VCC : 3.3V
- GND : Ground (0V)
- DO : Digital Out
  - You can inc/dec the threshold voltage by adjusting the potentiometer(십자 드라이버)
- AO : Analog Out

![1570964912024](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570964912024.png)

### Code

- 100ms 마다, 라이트 센서 Value를 읽어온다
- Light Status가 이전에서 바뀌었으면 Light turned on / Light turned off 출력
- 안 바뀌었으면, 메세지 안 띄운다

~~~python
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.IN)

current = None

while (True) :
    temp = GPIO.input(18)
    if (temp != current) :
        if (temp == 1) :
            print "Light turned off"
        else :
            print "Light turned on"
        current = temp
    time.sleep(0.1)
~~~

## Servo Motor

### Pin Description

- Orange : PWM input
- Red : VCC(typ : 4.8V ~ 6V)
- Brown : GND
- MG-90S

### 5V 베이스의 Servo를 3.3V GPIO에 연결하는 법

- 직접 연결(노이즈에 더 많이 노출됨)

  ![1570965101177](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570965101177.png)

- Source Driver / Level Converter 사용

  ![1570965112157](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570965112157.png)

  - 8채널 소스 드라이버 (TD62783APG)

    ![1570965134346](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570965134346.png)

- 트랜지스터 사용

  ![1570965151979](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570965151979.png)

### 서보 모터 제어

- PWM Signal 이용

  ![1570965176416](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570965176416.png)

- WiringPi Library offers a set of PWM API functions such as start(dutycycle) and ChangeDutyCycle(dutycycle), where dutycycle has a value ranging from 0 to 100(%)
  - 20ms PWM period에 대해, 1ms duty cycle(leftmost position, 0도)의 값은(%로)?
  - 20ms PWM period에 대해, 2ms duty cycle(rightmost position, 180도)의 값은(%로)?
  - 20ms PWM period에 대해, 1.5ms duty cycle(center position, 90도)의 값은(%로)?

~~~python
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT)

pos_center = 0 # Fill this using the answer for question 1

pwm = GPIO.PWM(25, 50)
pwm.start(pos_center)

dutycycle = input("Duty: ")
while dutycycle != 0:
    pwm.ChangeDutyCycle(dutycycle)
    dutycycle = input("Duty: ")
    
pwm.stop()
GPIO.cleanup()
~~~

- leftmost ~ rightmost로 smooth하게 sweep 하도록 할 수 있을까?
  - sweep speed 조절해보기
- 여러개의 서보를 동시에 제어하려면, pigpio나 servoblaster같은 라이브러리 사용하는 것을 추천
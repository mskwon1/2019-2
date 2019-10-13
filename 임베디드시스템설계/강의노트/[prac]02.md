### GPIO Pin Map

![1570963953261](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570963953261.png)

### Breadboard

- Used to build and test circuits quickly before finalizing any circuit design

![1570964041103](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570964041103.png)

- 각각 가로, 세로로 연결됨

### LED

![1570963907729](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570963907729.png)

### GPIO Commands

- T-shape PCB는 BCM넘버링 사용

  - gpio 커맨드를 사용할때 -g 옵션을 사용해줘야 함

  ~~~shell
  gpio readall # gpio 정보 전체 불러오기
  gpio mode 22 out # 22번 gpio의 mode를 out으로 설정
  gpio write 22 1 # 22번 gpio의 V를 1로 설정
  ~~~
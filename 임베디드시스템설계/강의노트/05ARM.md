# ARM

## ARM, Ltd.

- Acorn RISC Machine or Advanced RISC Machines
- Established in 1990
  - Spin-off from Acorn Computer
  - Founded by 12 engineers and CEO
  - Based in Cambridge, UK
  - Acquired by SoftBank, 2016
- Business areas
  - CPU Intellectual Properties
  - No chip manufacturing
  - Over 300 semiconductor partners

## ARM Architecture

- Programmer's model
  - Defines CPU architecture
  - Components
    - Intstruction set
    - Data access architecture
    - Operation mode
    - Register architecture
    - Exception handling mechanism
- An architecture has a number of CPU implementations
  - Around 20 CPU models are based on ARMv4 architecture

![1570367589653](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570367589653.png)

## ARM Instruction

- Data processing instruction

  ![1570952685077](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570952685077.png)

- LDR/STR

  ![1570952711372](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570952711372.png)

- Assembly Instructions

  ![1570952730105](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570952730105.png)

### Pipelines

![1570952772498](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570952772498.png)

### Instruction Set

- ARM Processor Instructions

  - 32 bits ARM instructions

    - Conditional execution
    - Load-Store, Branch instructions are using indirect address mode
    - 11 different instruction types in ARM mode instruction sets

    ![1570952874763](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570952874763.png)

  - 16 bits Thumb instructions

- Java Support

  - 16 bits Jazelle instruction-based
  - Hardware-based JVM processing optimizations

#### ARM/Thumb Interwork

- ARM state
  - ready to run 32 bits ARM instructions
  - when an exception occurs, 
    CPU state is changed to ARM state regardless of the current state
- Thumb state
  - 16 bits instructions sets are available
  - BX instruction changes CPU modes at run-time

#### Operating mode

- User mode
- FIQ(Fast Interrupt Request)
- IRQ(Interrupt Request)
- SVC
- Abort mode
- Undefined Mode
- System Mode

##### Mode Switch

- Exception

  - When exception occurs, HW automatically switches the operating mode to appropriate one

    - When FIQ occurs, HW automatically switches the operating mode to FIQ mode and runs the exception handler

  - Exception types tightly correspond to operating mode

  - Exception Vectors

    ![1570953100476](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570953100476.png)

- System calls
  - System calls are implemented by 'SWI' instruction in ARM
  - HW automatically switches the operating mode from USER to SVC

#### ARM Registers

- General purpose registers

  - 30 registers for data processing instructions

- Special registers

  - PC : R15
  - Current Program Status Register(CPSR)
  - Saved Program Status Register(SPSR)
    - saves CPSR of immediately preceding operating mode when mode switch occurs

- ARM registers for operating modes

  ![1570953210380](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570953210380.png)

- Thumb Mode Registers

  ![1570953225918](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570953225918.png)

##### Stack Pointer(R13)

- A general purpose register dedicated  for another special purpose
- Stores the current SP position
- Each operating mode has it's own SP register
- ARM does not provide PUSH/POP instructions
  - Stack operations are implemented using LDM/STM instructions

##### Link Register(LR or R14)

- Stores return address when subroutine call occurs

  - BL instruction automatically store the return address to LR

  - Moving the data stored in LR to PC leads to returning from subroutine

    ![1570953366845](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570953366845.png)

  - The address stored in LR can be modified using data processing instructions

- Each operating mode has it's own LR register

##### Program Counter (PC or R15)

- Stores the current address to execute instructions
- The contents of PC can be directly modified using data processing instructions
- Only one PC register in a system

##### Program Status Register

- PSR Registers in ARM

  - 1 CPSR
  - 5 SPSR(for each operating mode)

- PSR Register Fields

  ![1570953486937](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1570953486937.png)

  - Condition Flag

    - reflects ALU processing results

  - Control Bits

    - To control CPU Status

    
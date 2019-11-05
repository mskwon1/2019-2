# Greedy Algorithms

- 각 단계에서 가장 좋을거라 생각되는 선택을 취함
- 반드시 최적의 해를 구한다고 보장할 수는 없다
  - Greedy-choice property를 가지는 경우에만 최적해를 구한다
- ex)
  - activity selection
  - huffman code
  - minimum spanning tree algorithms
  - dijkstra's algorithm for shortest paths from a single source
  - rod-cutting : dynamic으로 풀어야함

## An Activity-Selection Problem

- Activity들이 finish time의 **단조 증가 순으로 정렬**되어 있을 때

  ![1571284656304](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571284656304.png)

- 활동 시간이 겹치지 않게 compatible activites의 최대 집합을 찾는 문제

  ![1571284671466](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571284671466.png)

- The activity selection problem exhibits optimal substructure

  ![1571284706398](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571284706398.png)

  - The greedy choice : $a_i$ with smallest $f_i$ = the first activity is always in the solution

    ![1571284765478](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571284765478.png)

### Recursive Activity Selector

- Returns a maximum-size set of mutually compatible activities in $S_k$

  ![1571284825203](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571284825203.png)

  ![1571284847489](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571284847489.png)

### Iterative Function

![1571284872518](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571284872518.png)

## Elements of Greedy Algorithms

- Greedy algorithm 만들기
  - 하나의 (grreedy) 선택을 하면 나머지 부분도 하나의 subproblem만 남도록 최적화 문제를 세워라
  - Prove that there is always an optimal solution to the original problem that makes the greedy choice
  - Demonstrate optimal structure(Greed choice와 subproblem의 optimal solution을 결합하면 전체 문제의 optimal solution을 얻는다는 것을 보임)
- When can we use a greedy algorithms?
  - greedy-choice property + optimal substructure

### Greedy-Choice Property

- 어떤 선택을 할지 고려할 때 부분 문제들의 결과를 고려할 필요없이 현재 고려 중인 문제에서 최적인 문제를 선택해도 된다
  - dynamic programming : subproblem들의 해를 먼저 구한다 (bottom-up)
  - greedy algorithm : choice를 먼저 한 다음 나머지 subproblem을 푼다 (top-down)
    - ex) rod-cutting problem은 greedy-choice property가 없다 -> dynamic으로 풀어야 함

### Optimal Substructure

- An optimal solution to the problem contains optimal solutions to subproblems

## 0-1 Knapsack Problem

- n items : $w_i$ is a weight of $i$-th item, $v_i$ is a value of $i$-th item

- $W$ : knapsack capacity

- problem : choose a set of items maximizing total value, and not exceeding knapsack capacity

  ![1571285178827](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571285178827.png)

- Has optimal substructure but not greedy-choice property -> dynamic programming

### Fractional knapsack problem

- n items : $w_i$ is a weight of $i$-th item, $v_i$ is a value of $i$-th item

- $W$ : knapsack capacity

- problem : choose a set of fractional items maximizing total value, and not exceeding knapsack capacity

  ![1571285218793](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571285218793.png)

- Has optimal substructure and greedy-choice property -> greedy algorithm $O(n\ lg\ n)$

  - sort items in value / wieght ($=v_i/w_i$)

    ![1571285327175](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571285327175.png)

## Huffman Codes(for data compression)

![1571285458858](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571285458858.png)

- a~f 문자를 100000개 포함한 파일의 길이
  - fixed-length (3-bit) code 사용 : 300000 비트 필요
  - variable-length code 사용 : 파일 크기 줄이기 가능(가변적) - 224000 비트 필요
    - codeword의 끝을 어떻게 알 것인가

### Prefix codes

![1572856829263](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572856829263.png)

- 어느 codeword도 다른 codeword의 prefix가 아니다
  - guarantees unambiguity in decoding

### Problem

- 주어진 문자 분포에 대해 B(T)를 최소화하는 최적의 Prefix Code를 만들어라

  $B(T) = \sum_{c\in C}c.freq*d_T(c)$

  - $C$ : 파일에 사용된 문자 집합

  - $c.freq$ : 문자 $c$가 파일에서 사용된 빈도 (사용된 횟수 / 전체 문자 갯수)

  - $d_T(c)$ : 만들어진 code tree에서 문자 c를 나타내는 노드의 depth(root 에서부터 edge의 갯수)

    ![1572857087521](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572857087521.png)

    - 최적의 prefix code는 항상 full binary tree로 표현됨(자식이 0개 또는 2개)

![1572857130636](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572857130636.png)

![1572857184557](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572857184557.png)

![1571285495078](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1571285495078.png)

### Optimizing prefix code with respect to $B(T)$

$B(T) = \sum_{c\in C}c.freq*d_T(c)$

- $C$ : 파일에 사용된 문자 집합

- $c.freq$ : 문자 $c$가 파일에서 사용된 빈도 (사용된 횟수 / 전체 문자 갯수)

- $d_T(c)$ : 만들어진 code tree에서 문자 c를 나타내는 노드의 depth(root 에서부터 edge의 갯수)
  = codeword의 길이

![1572857282861](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572857282861.png)

#### Lemma 16.2

- Let C be an alphabet in which each character $c\in C$ has frequency $c.freq$
- Let $x$ and $y$ be two characters in $C$ having the lowest frequencies
- Then there exists an optimal prefix code for $C$ in which the codewords for $x$ and $y$ have the same length and differ only in the last bit

##### 증명

- 주어진 문제에 대한 임의의 optimal prefix code를 나타내는 tree T 를 변형하여 x와 y의 최대 깊이를 갖는 sibling leaf node가 되는 $T''$를 만들면 $T''$도 optimal prefix code를 나타냄을 보임
  - $B(T) = B(T'')$임을 보임

![1572857446690](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572857446690.png)

![1572857477287](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572857477287.png)

![1572857529149](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572857529149.png)

#### Lemma 16.3

- Let C be an alphabet in which each character $c\in C$ has frequency $c.freq$

- Let $x$ and $y$ be two characters in $C$ having the lowest frequencies

- Let $C' = C - \{x,y\} \cup \{z\}$. In $C'$

- $z.freq = x.freq + y.freq$ and $c.freq$ are same as in $C$ for all other characters

- Let $T'$ be any tree representing an optimal prefix code for $C'$

- Then the tree $T$, obtained from $T'$ by replacing the leaf node for $z$ with an internal node having $x$ and $y$ as children , represents an optimal prefix code for the alphabet $C$

  ![1572857698502](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572857698502.png)

![1572857725456](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1572857725456.png)

- Lemma 16.2, 16.3에 의해 Huffman code algorithm은 optimal prefix code를 만든다
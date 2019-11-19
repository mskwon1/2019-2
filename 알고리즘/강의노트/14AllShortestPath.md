# All-Pairs Shortest Paths

### Weight Matrix Representation

- Representation of weight matrix W in G = (V, E)

![image-20191119120527835](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119120527835.png)

### All-Pairs Shortest Paths

- Problem of finding shortest paths between all pairs of vertices in a graph (with **negative edges**, but **no negative-weight cycle**)

- Solutions represented with

  - distance matrix $D$ where $d_y = \delta(i,j)$
  - predecessor matrix $\Pi$ where $pi_{ij}$ : predecessor of $j$ on some shortest path $i$

  ![image-20191119121005961](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119121005961.png)

  ![image-20191119121156933](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119121156933.png)

- Simple solution : $V$ repetition of  single-source shortest paths algorithm
  - $V$ x Bellman-Ford algorithm : $V$ x $O(VE)$ = $O(V^2E)$ = $O(V^4)$ in dense graphs
  - $V$ x Dijkstra's algorithm : $V$ x $O(E\ lg\ V)$ = $O(VE\ lg \ v)$ 
    or $V$ x $O(V^2+E)$ = $O(V^3+VE)$ = $O(V^3)$
- 2 dynamic programming alogrithms
  - Using matrix multiplication : $\theta(V^3\ lg\ V)$
  - Floyd-Warshall algorithm : $\theta(V^3)$

![image-20191119122620560](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119122620560.png)

#### Dijkstra's Algorithm

- used when $G$ has no negative edges
- Greedy Algorithm

![image-20191119123006894](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119123006894.png)

## With Matrix Multiplication

- Let $l_{ij}^{(m)}$ be the minimum weight of any path from vertex $i$ to $j$ that contains at most $m$ edges

  ![image-20191119123109928](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119123109928.png)

  - shortest path는 최대 $n-1$ edges를 가지므로 $\delta(i,j) = l_{ij}^{(n-1)}$

- Taking as our input the matrix $W = (w_{ij})$, we now compute a series of matrices $L^{(1)}, ... , L^{(n-1)}$, where for $m$ = 1, 2, ..., $n-1$, we have $L^{(m)} = (l_{ij}^{(m)})$

### O(n^4)

![image-20191119123335872](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119123335872.png)

![image-20191119123339892](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119123339892.png)

![image-20191119123354941](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119123354941.png)

![image-20191119123422394](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119123422394.png)

![image-20191119123443016](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119123443016.png)

![image-20191119123909845](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119123909845.png)

![image-20191119123938383](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119123938383.png)

- Observation : The matrix multiplication is associative

![image-20191119124009677](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119124009677.png)

### O(n^3 log n)

![image-20191119124513226](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119124513226.png)

![image-20191119124516105](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119124516105.png)

## Floyd-Warshall Algorithm

- an **intermediate** vertex of a simple path $p = \{v_1, v_2, ..., v_l\}$ is any vertex of $p$ other than $v_1$ or $v_l$, that is, any vertex in the set = $\{v_2, ... v_{l-1}\}$

- For any pair of vertices $i, j$ in $V$, consider all paths from $i$ to $j$ whose intermediate vertices are all drawn from $\{1, 2, ..., k\}$ and let $p$ be a minimum-weight path from among them
- $O(n^3)$

![image-20191119125221800](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119125221800.png)

- $k = 0$

  ![image-20191119125353592](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119125353592.png)

- $k = 1$

  ![image-20191119125415959](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119125415959.png)

- $k=2$

  ![image-20191119125442319](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119125442319.png)

- $k=3$

  ![image-20191119125455602](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119125455602.png)

- $k=4$

  ![image-20191119125507198](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119125507198.png)

- $k=5$

  ![image-20191119125520245](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191119125520245.png)
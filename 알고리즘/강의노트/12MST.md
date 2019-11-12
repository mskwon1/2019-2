# Minimum Spanning Trees

- Given an undirected **weighted** graph $G = (V, E)$

- spanning tree $G_s = (V,E_s)$ where $E_s$ is a subset of $E$ that connects all the nodes in $G$

- minimum spanning tree : spanning tree with the minimum total weight

  $w(T) = \sum_{(w,v)\in T}w(u,v)$

![image-20191112120928869](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191112120928869.png)

- An undirected weighed graph and its minimum spanning tree

  ![image-20191112120946366](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191112120946366.png)

### MST Algorithm

- safe edge : For an edge set $A$ which is a subset of some MST, if $A \cup e$ is still a subset of a MST, then $e$ is a **safe** edge

- loop invariant in GENERIC-MST algorithm:

  - prior to each iteration, A is a subset of some MST

  ![image-20191112121559716](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191112121559716.png)
  - spanning tree가 될 때까지 safe edge만 계속 더하기

#### Theorem 23.1

- Connected undirected weighted graph $G$에 대해서, edge set $A$는 $G$의 한 MST의 부분집합이라 하자. $A$를 존중(**respect**)하는 $G$의 cut $(S, V-S)$가 있고, $(u,v)$가 $(S, V-S)$를 cross 하는 light edge 라면 $(u,v)$는 $A$에 대한 **safe edge**이다

  ![image-20191112121818922](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191112121818922.png)

  - light edge : cross edge 중에서 weight가 가장 작은 edge

- Proof

  - $A$를 포함하는 MST를 $T$라 하자
    - $A \cup (u,v)$가 $T$에 포함되면, $(u,v)$는 safe edge : trivial
    - $A \cup (u,v)$가 $T$에 포함되지 않으면, $T$가 spanning tree 이므로 $T$안에 $u$ :arrow_forward: $v\ path\ p$가 있고 그 path 에는 cross edge가 있다
    - 이 cross edge를 $(x,y)$라 하고 이것을 제거하면 $T$는 더 이상 connected가 아니고 다시 $(u,v)$를 추가하면 $T' = T - \{(x,y)\} \cup \{(u,v)\}$는 spanning tree가 되는데 $(u,v)$가 light edge $w(u,v) \le w(x,y)$이므로 이 $w(T') = w(T) - w(x,y) + w(u,v) \lt w(T)$이다
    - $T$가 MST이므로 $w(T') = w(T)$ 즉, $T'$도 MST

#### MST-KRUSKAL

![image-20191112123321161](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191112123321161.png)

![image-20191112123604926](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191112123604926.png)

![image-20191112123619972](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191112123619972.png)

#### Prim's Algorithm

- Main idea

  - Maintain a set $S$ that starts out with a single node $s$
  - Find the smallest weighted edge $e^* = (u,v)$ that connects $u \in S$ and $v \in S$
  - Add $e^*$ to the MST, add $v$ to $S$
  - Repeat until $S = V$

- Differs from Kruskal's in that we grow a single supernode $S$ instead of growing multiple ones at the same time

  ![image-20191112125158281](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191112125158281.png)

  ![image-20191112125442867](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191112125442867.png)

![image-20191112125450244](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191112125450244.png)
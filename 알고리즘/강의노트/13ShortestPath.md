# Single-Source Shortest Paths

## Problem

The problem of finding shortest paths from a source vertex $s$ to other vertices in the graph

![image-20191114122022529](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114122022529.png)

- Bellman-Ford algorithm : works in a graph with negative weights
- Dijkstra's algorithm : works in a graph with non-negative weights

### Variants

- Single destination shortest-paths problem
  - edge 방향을 반대로 하고 single-source shortest-paths problem을 푼다
- Single-pair shortest-path problem
  - single-source shortest paths problem을 풀면 그 안에 해가 포함되어 있다.
  - single-pair shortest path problem만 푸는 알고리즘의 worst-case running time은 가장 좋은 single-source shortest-paths problem의 worst-case running time과 점근적으로 같다
- All-pairs shortest-paths problem
  - 모든 vertices에 대해 single-source shortest-paths problem을 푼다.
    그러나 Floyd-Warshall algorithm과 같이 더 효율적인 방법도 있다

### Optimal substructure of a shortest path

![image-20191114122349823](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114122349823.png)

Thus, Dijkstra's algorithm : greedy algorithm, Floyd-Warshall algorithm : dynamic programming

#### Negative-weight edges

![image-20191114122722278](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114122722278.png)

- Graph $G$에 negative-weight **cycle**이 있으면 shortest-path problem은 well-defined가 아니다
  - s -> e, s -> f, s -> g 때문
- Bellman-Ford algorithm 
  - works in a graph with **negative weights**, unless there is a **negative cycle**
- Dijkstra's algorithm
  - works in a graph with **non-negative weights**

#### Shortest-paths tree

![image-20191114123234787](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114123234787.png)

- **predecessor subgraph** defined by "$v.\pi$ : v의 predecessor in shortest-paths tree"와 같다

#### Relaxation

Update operation of $v.d(shortest-path\ estimate)$

![image-20191114123516996](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114123516996.png)

![image-20191114123521041](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114123521041.png)

## Bellman-Ford Algorithm

- Single source shortest path algorithm

- Unlike Dijkstra's algorithm, edges can have negative weight

- The algorithm returns false when there is a negative cycle

![image-20191114124101054](C:\Users\user\Desktop\2019-2\알고리즘\강의노트\image-20191114124101054.png)

![image-20191114124207674](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114124207674.png)

## Topological sort를 이용한 single-source shortest path

- $G$가 directed acyclic graph일 때(cycle이 없으므로 negative-weight cycle도 없음) 사용 가능

  ![image-20191114124319746](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114124319746.png)

![image-20191114124324653](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114124324653.png)

![image-20191114124348667](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114124348667.png)

## Dijkstra's Algorithm

- used when $G$ has no negative edges

- Greedy Algorithm

  ![image-20191114124421707](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114124421707.png)

  ![image-20191114124433701](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114124433701.png)

![image-20191114124447631](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191114124447631.png)


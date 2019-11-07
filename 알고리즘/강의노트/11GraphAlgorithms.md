# Graph Algorithms

## Graphs

![image-20191105120926473](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105120926473.png)

- An Abstract way of representing connectivity using **nodes**(also called **vertices**) and **edges**

- $m$ edges connect some pairs of nodes
  - Edges can be either directed or undirected
- Nodes and edges can have some auxiliary(보조의) information

### Definitions

- Undirected Graph $G$

  ![image-20191105121218643](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105121218643.png)

  - A pair $(V,E)$, where $V$ is a a finite set of points called vertices and $E$ is a finite set of edges

  - can be thought of as a directed graph

    ![image-20191105121228225](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105121228225.png)

- Directed Graph

  ![image-20191105121241899](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105121241899.png)

  - The edge $e$ is an **ordered** pair $(u,v)$
  - An edge $(u,v)$ is **incident from** vetex $u$ and is **incident to** vertex $v$

- A **path** from a vertex $v$ to a vertex $u$ is a sequence $(v_0, v_1, v_2, ... , v_k)$ of vertices where $v_0 = v, v_k = u$ and $(v_i, v_{i+1}) \in E$ for $i = 0, 1, ..., k-1$
- A vertex $u'$ is **reachable**  from vertex $u$ if there is a path $p$ from $u$ to $u'$ in $G$
- The **length of path** is defined as the number of edges in the path
- A **cycle** is a path where $v_0 = v_k$
- An undriected graph is **connected** if every pair of vertices is connected by a path
- A **forest** is an acyclic(cycle이 없는) graph(tree가 여러개)
  , and a **tree** is a connected acyclic graph
- A graph taht has weights associated with each edge is called a **weighted graph**

### Tree

- A connected acyclic graph
- Most important type of special graphs
  - many problems are easier to solve on trees
- Alternate equivalent definitions
  - A connected graph with $n \gt 1$ edges (where $n$ is a number of vertices)
  - An acyclic graph with $n \gt 1$ edges
  - There is exactly **one path** between every pair of nodes
  - An acyclic graph but adding any edge results in a cycle
  - A connected graph but removing any edge disconnects it

### Representation of a graph

- Graphs can be represented by their adjacency matrix or adjacency list
- Adjacency matrices have a value $a_{ij} = 1$ if nodes $i$ and $j$ share an edge; 0 otherwise.
  In case of a weighted graph, $a_{ij} = w_{ij}$, the weight of the edge
- The adjacency list representation of a graph $G = (V, E)$ consists of an array $Adj[1...|V|]$ of lists. Each list $Adj[v]$ is a list of all vertices adjacent to $v$
- For a graph with $n$ nodes, adjacency matrices take $\theta(n^2)$ space and adjacency list takes $\theta(|E|+|V|)$ space
- 교과서에서는 대부분 **Adjacency List** 표현을 가정

#### Undirected Graph

![image-20191105122412119](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105122412119.png)

#### Directed Graph

![image-20191105122430408](C:\Users\user\Desktop\2019-2\알고리즘\강의노트\image-20191105122430408.png)

### Graph Traversal

- The most basic graph algorithm that visit nodes of a graph in certain ored
- Used as a subroutine in many other algorithms
  - 특별한 언급이 없으면 vertex는 **알파벳 순**으로 처리

#### Breadth-First Search(BFS) : uses queue

![image-20191105122913793](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105122913793.png)

- vertex $s$로부터 reachable vertex $v$에 대한 shortest path distance $\delta(s,v)$를 모두 계산한다
  그 값은 $v.d$이다

  <img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105123046811.png" alt="image-20191105123046811" style="zoom:80%;" />

#### Depth-First Search(DFS) : uses recursion(stack)

![image-20191105123344693](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105123344693.png)

- 앞은 Starting Time, 뒤는 Finish Time(back tracking)

##### DFS Forest

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105123710355.png" alt="image-20191105123710355" style="zoom:80%;" />

##### DFS Algorithm

![image-20191105123738889](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105123738889.png)

![image-20191105123857220](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105123857220.png)

- Backward edge : 돌아가는 edge
- Cross edge : 서로 다른 트리에서 access 하는 경우
- Forward edge : 트리 내에서 선택되지 않은 edge?
- Tree edge : 트리에 포함되는 edge

##### Properties

- 다음과 같은 timestamped directed graph가 만들어진다

  ![image-20191105123925239](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105123925239.png)

### Topological Sort

- DAG : Directed Acyclic Graph(사이클이 없는 Directed Graph)
  - Total Order가 있는 집합
    - Sorting 그냥 하면 됨
  - **Partial Order가 있는 집합**(위의 예)
    - 대소관계가 있는 것들끼리 관계를 그래프로 표현

![image-20191105130224212](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105130224212.png)

- Finish Time에 따라서 리스트의 앞에 집어넣는다

#### Theorem 22.12

- TOPOLOGICAL-SORT algorithm produces a topological sort of the directed acyclic graph provided as its input

  ![image-20191105131245825](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105131245825.png)

![image-20191105131256406](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191105131256406.png)

### Strongly connected components

- Undirected graph
  - **connected** if every vertex is reachable from all other vertices
  - **connected components** of a graph are the equivalence classes of vertices under the 'is reachable from' relation
- Directed graph
  - **strongly connected** if every two vertices are reachable from each other
  - **strongly connected components** of a directed graph are the equivalence classes of vertices under the 'are mutually reachable' relation

![image-20191107125252490](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191107125252490.png)

![image-20191107125309781](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191107125309781.png)

#### DFS(G)

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191107125502780.png" alt="image-20191107125502780" style="zoom: 80%;" />

#### Compute $G^T$

- All directions are reveresd

  ![image-20191107125540937](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191107125540937.png)

#### DFS($G^T$)

- Vertices are selected in order of decreasing u.f

  ![image-20191107125609571](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191107125609571.png)

#### SCC

![image-20191107125645792](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20191107125645792.png)

### Euler Tour

- An Euler tour of a strongly connected, directed graph $G = (V,E)$ is a cycle that traverses each edge of $G$ exactly once, although it may visit a vertex more than once
  - $G$ has an Euler tour iff in-dgree(v) = out-degree(v) for all $v \in V$
  - The Euler tour algorithm runs in $O(E)$
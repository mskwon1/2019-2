## String Matching

- Substring search

  - Find a pattern of length M in a text of length N
    (typically N >> M)

    <img src="../../typora_images/15StringMatching/image-20191126120918758.png" alt="image-20191126120918758" style="zoom:67%;" />

### Brute-Force Substring Search

#### Naive algorithm

- Check for pattern starting at each text position

  <img src="../../typora_images/15StringMatching/image-20191126121001531.png" alt="image-20191126121001531" style="zoom:67%;" />

  <img src="../../typora_images/15StringMatching/image-20191126121053792.png" alt="image-20191126121053792" style="zoom:67%;" />

  <img src="../../typora_images/15StringMatching/image-20191126121106021.png" alt="image-20191126121106021" style="zoom:67%;" />

  - can be slow if text and pattern are repetitive

- Improvement

  - develop a linear time algorithm
  - avoid **backup**
    - naive algorithm needs backup for every mismatch
    - thus naive algorithm cannot be used when input text is a stream

  ![image-20191126121237581](../../typora_images/15StringMatching/image-20191126121237581.png)

### Knuth-Morris-Pratt(KMP) Algorithm

- Clever method to always avoid **backup** problem

  <img src="../../typora_images/15StringMatching/image-20191126121356096.png" alt="image-20191126121356096" style="zoom:67%;" />

### Deterministic Finite Automaton

- DFA

  - Finite number of states (including **start** and **accept states**)
  - Exactly one transition for each char
  - Accept if sequence of transitions leads to accept state

  <img src="../../typora_images/15StringMatching/image-20191126121607409.png" alt="image-20191126121607409" style="zoom:67%;" />

  <img src="../../typora_images/15StringMatching/image-20191126121636357.png" alt="image-20191126121636357" style="zoom:67%;" />

  <img src="../../typora_images/15StringMatching/image-20191126121903289.png" alt="image-20191126121903289" style="zoom:67%;" />

  <img src="../../typora_images/15StringMatching/image-20191126121713282.png" alt="image-20191126121713282" style="zoom:67%;" />

- Difference from naive algorithm

  - precomputation of DFA[][] from pattern
  - text pointer `i` never decrements (**no backup**)

  <img src="../../typora_images/15StringMatching/image-20191126121828233.png" alt="image-20191126121828233" style="zoom:67%;" />



- The state of DFA represents

  - the number of characters in pattern that have been matched

    <img src="../../typora_images/15StringMatching/image-20191126121937186.png" alt="image-20191126121937186" style="zoom:67%;" />
  
- Prefix / Sufix of a Text

  <img src="../../typora_images/15StringMatching/image-20191126145844870.png" alt="image-20191126145844870" style="zoom:67%;" />

  

#### DFA Construction

- Suppose that all transitions from state `0` to stat `j-1` are already computed

- Match transition

  - If in state `j` and next char `char c = pattern[j]`, then transit to state `j+1`

    <img src="../../typora_images/15StringMatching/image-20191126145908290.png" alt="image-20191126145908290" style="zoom:67%;" />

- Mismatch transition

  - If in state `j` and next`char c != pattern[j]`, then which state to transit

    <img src="../../typora_images/15StringMatching/image-20191126145931891.png" alt="image-20191126145931891" style="zoom:67%;" />

  <img src="../../typora_images/15StringMatching/image-20191126122130057.png" alt="image-20191126122130057" style="zoom:67%;" />

  - then the last `j-1` characters of input text are 
    `pattern[1] ~ pattern[j-1]`, followed by `c`
  - to compute `DFA[c][j]`:
    - simulate `pattern[1] ~ pattern[j-1]` on DFA (still under construction) and let the current state `X`
    
    - Then `DFA[c][j] = DFA[c][X]`
    
      <img src="../../typora_images/15StringMatching/image-20191126150032122.png" alt="image-20191126150032122" style="zoom:67%;" />
    
    - take a transition `c` from state `X`
    
    - Running time : require `j` steps
    
    - **But, if we maintain state X, it takes only constant time!**

- Maintaining state `X`:

  - finished computing transitions from state `j`

  - Now, move to next state `j+1`

  - then what the new state(`X'`) of `X` be?

    <img src="../../typora_images/15StringMatching/image-20191126122650961.png" alt="image-20191126122650961" style="zoom:67%;" />

- A Linear Time Algorithm

  - for each state `j`

    - Match case : set `DFA[pattern[j]][j] = j+1`
    - Mismatch case : copy `DFA[][X]` to `DFA[][j]`
    - Update `X`

    <img src="../../typora_images/15StringMatching/image-20191126122801136.png" alt="image-20191126122801136" style="zoom:67%;" />

- Example

  <img src="../../typora_images/15StringMatching/image-20191126122821449.png" alt="image-20191126122821449" style="zoom:67%;" />

  <img src="../../typora_images/15StringMatching/image-20191126122831159.png" alt="image-20191126122831159" style="zoom:67%;" />
  
  <img src="../../typora_images/15StringMatching/image-20191126150131526.png" alt="image-20191126150131526" style="zoom:67%;" />

#### Algorithm with DFA

- String matching algorithm with DFA accesses no more than M+N chars to search for a pattern of length M in a text of length N


- `DFA[][]` can be constructed in time and space of order `O(RM)`, where `R` is the number of characters used in a text
- Questions : Text에 나타나는 모든 pattern을 찾을 수 있는가?

  - Text : AAAAAAAAA
  - Pattern : AAAAA
  - Solution : 0, 1, 2, 3, 4, 5

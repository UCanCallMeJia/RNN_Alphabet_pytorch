# RNN_Alphabet_pytorch
An interesting simple for predicting alphabet.
# Introduction
  代码中定义了两种模型，一种使用了RNN框架，另一种使用LSTM；  
  两种框架结果不同；
  1. 使用`RNN`的结果：
  ```python
Input: ['N', 'O', 'P', 'Q', 'R']  Prediction: S
Input: ['G', 'H', 'I', 'J', 'K']  Prediction: L
Input: ['L', 'M', 'N', 'O', 'P']  Prediction: Q
Input: ['M', 'N', 'O', 'P', 'Q']  Prediction: R
Input: ['B', 'C', 'D', 'E', 'F']  Prediction: F
Input: ['C', 'D', 'E', 'F', 'G']  Prediction: H
Input: ['J', 'K', 'L', 'M', 'N']  Prediction: O
Input: ['H', 'I', 'J', 'K', 'L']  Prediction: M
Input: ['S', 'T', 'U', 'V', 'W']  Prediction: X
Input: ['K', 'L', 'M', 'N', 'O']  Prediction: P
Input: ['I', 'J', 'K', 'L', 'M']  Prediction: N
Input: ['F', 'G', 'H', 'I', 'J']  Prediction: K
Input: ['Q', 'R', 'S', 'T', 'U']  Prediction: V
Input: ['P', 'Q', 'R', 'S', 'T']  Prediction: U
Input: ['D', 'E', 'F', 'G', 'H']  Prediction: I
Input: ['U', 'V', 'W', 'X', 'Y']  Prediction: Z
Input: ['R', 'S', 'T', 'U', 'V']  Prediction: W
Input: ['E', 'F', 'G', 'H', 'I']  Prediction: J
Input: ['A', 'B', 'C', 'D', 'E']  Prediction: F
Input: ['O', 'P', 'Q', 'R', 'S']  Prediction: T
Input: ['T', 'U', 'V', 'W', 'X']  Prediction: Z
 
  ```
  2. 使用`LSTM`的结果：
  ```python
Input: ['C', 'D', 'E', 'F', 'G']  Prediction: H
Input: ['U', 'V', 'W', 'X', 'Y']  Prediction: Z
Input: ['J', 'K', 'L', 'M', 'N']  Prediction: O
Input: ['G', 'H', 'I', 'J', 'K']  Prediction: L
Input: ['D', 'E', 'F', 'G', 'H']  Prediction: I
Input: ['N', 'O', 'P', 'Q', 'R']  Prediction: S
Input: ['L', 'M', 'N', 'O', 'P']  Prediction: Q
Input: ['F', 'G', 'H', 'I', 'J']  Prediction: K
Input: ['R', 'S', 'T', 'U', 'V']  Prediction: W
Input: ['O', 'P', 'Q', 'R', 'S']  Prediction: T
Input: ['K', 'L', 'M', 'N', 'O']  Prediction: P
Input: ['H', 'I', 'J', 'K', 'L']  Prediction: M
Input: ['Q', 'R', 'S', 'T', 'U']  Prediction: V
Input: ['M', 'N', 'O', 'P', 'Q']  Prediction: R
Input: ['I', 'J', 'K', 'L', 'M']  Prediction: N
Input: ['P', 'Q', 'R', 'S', 'T']  Prediction: U
Input: ['B', 'C', 'D', 'E', 'F']  Prediction: G
Input: ['S', 'T', 'U', 'V', 'W']  Prediction: X
Input: ['E', 'F', 'G', 'H', 'I']  Prediction: J
Input: ['A', 'B', 'C', 'D', 'E']  Prediction: F
Input: ['T', 'U', 'V', 'W', 'X']  Prediction: Y
  ```
  **思考：为什么LSTM性能更好？ 改变序列长度，会怎样？**
  
  Contact: jiazx@buaa.edu.cn

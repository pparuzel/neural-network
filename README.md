# neural-network
Everything related to the neural networks I really need to explore

To test **fifa/nn.py** with predefined weights and bias - type:
```
python3 nn.py '[0.7367558056631766, 0.45169329115720835]' 13.96682537688979
```

I actually worked out the derivatives for *Gradient Descent* myself so the formulas might sometimes occur invalid.  
  
## Predicting stamina of Fifa players
#### TO-BE-DONE: Use MLP. Single Perceptron is not enough to be accurate.
**fifa/nn.py** is a single-layer Neural Network.  
 
For activation function I used *ReLU* so for some random weights a restart will be needed as the net will most likely overshoot the minima.  
Error function is as simple as possible. *Squared loss function* was applied.  
  
## Predicting sum of two values
Same was put into **sumation/nn.py**. The goal of this NN is to predict the sum of two values.  
  
## Predicting result of x XOR y = z
#### Purpose: To show that Single-perceptron NN is insufficient to find perfect solution.
**xor_problem/nn.py** - super simple XOR Problem that single-layer perceptron is unable to resolve.  
**MLP** (MultiLayer Perceptron) is required.  

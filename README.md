# BNN_basic
Bayesian neural network implementation using multi-layer perceptron (2 layers) and independent gaussian prior on each weight. Tested on a small dataset for binary class classification. 

The Gaussian prior on each weight ${tex`w_i`}is ${tex`\prod_i N(w_i|\mu_ilog(1+exp(p_i)))`} where ${tex`\mu_i`} and ${tex`\p_i`} are variational parameters. 

# Result
train accuracy:  0.93828005
test accuracy:  0.93263996
within 1000 epochs

See the figures for visualization

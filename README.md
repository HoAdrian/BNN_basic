<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# BNN_basic
Bayesian neural network implementation using multi-layer perceptron (2 layers) and independent gaussian prior on each weight. Tested on a small dataset for binary class classification. The dataset consists of 872 training examples and 500 testing examples. 

The Gaussian prior on each weight $w_i$ is $$q(w_i) = N(w_i|\mu_i*\log(1+\exp(p_i)))$$ where $\mu_i$ and $p_i$ are variational parameters.

The function to be maximized is the variational lower bound:
![var_low_bound](https://github.com/HoAdrian/BNN_basic/blob/main/images/var_low_bd.png)
<!-- $$ L(w) = \sum_{n=1}^N E_{q(w)}\[\log(p(y_n|f_w(x_n)))\] - KL(q(w)||p(w))$$ -->
where $f_w$ is the neural network. 



# Result
train accuracy: 0.9635757, 
test accuracy:  0.961684, 
within 200 epochs using a minibatch size of 100. 

See the figures for visualization

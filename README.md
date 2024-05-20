<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# BNN_basic
Bayesian neural network implementation using multi-layer perceptron (2 layers) and independent gaussian prior on each weight. Tested on a small dataset for binary class classification. 

The Gaussian prior on each weight $w_i$ is $$q(w_i) = N(w_i|\mu_i*\log(1+\exp(p_i)))$$ where $\mu_i$ and $p_i$ are variational parameters.

The function to be maximized is the variational lower bound:
![var_low_bound](https://github.com/HoAdrian/BNN_basic/blob/main/images/var_low_bd.png)
<!-- $$ L(w) = \sum_{n=1}^N E_{q(w)}\[\log(p(y_n|f_w(x_n)))\] - KL(q(w)||p(w))$$ -->
where $f_w$ is the neural network. Since the dataset is pretty small, I did not split the dataset into minibatches. 



# Result
train accuracy:  0.93828005, 
test accuracy:  0.93263996, 
within 1000 epochs. 

See the figures for visualization

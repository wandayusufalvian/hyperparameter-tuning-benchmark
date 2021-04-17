import numpy as np
from scipy.stats import uniform
from skopt.space import Real,Integer
print(Real(0.001,1,'log-uniform'))
print(uniform(loc=0, scale=4))

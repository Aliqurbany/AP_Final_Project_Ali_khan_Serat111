import pandas as pd 
import config
import numpy as np
from preprocessing import add_features

conf = type("cfg", (), {})()
config.FEATURE_COLS = ['temp']

df = pd.DataFrame({'temp': [10, 12, 11, 15, 144, 14, 20]})

out = add_features(df, win_std=3, win_ma=4)
print(out)

#iput bacht = 2 features = 10
x = np.random.rand(2, 10)

# Dense layer with 32 unites 
W = np.random.rand(10, 32)
# biase 
b = np.random.rand(32)

#Linear transform 

y = np.dot(x,W)*b
print(x.shape)
print(W.shape)
print(b.shape)
print(y.shape) # each sample represented in 32 dimentions

# Now add activation function for nonlinearity 

# context vector computation 

# suppose you have 4 timesteps each with 3 features 

X = [
  [1.0, 0.5, 0.2],   # t1
  [0.9, 0.4, 0.1],   # t2
  [2.0, 1.5, 0.3],   # t3
  [1.1, 0.7, 0.2]    # t4
]

# you apply softmax activation function and get these numbers
# attention weights (softmax) the most important information in that timesteps 
#a = [0.05, 0.10, 0.60, 0.25]

# context = a*b 

#context = 0.05*[1.0,0.5,0.2] +0.10*[0.9,0.4,0.1] +0.60*[2.0,1.5,0.3] + 0.25*[1.1,0.7,0.2]

# result: 
#context = [1.55, 1.12, 0.25]
# so instead of 4 timesteps x 3 features you now have one 3-dim vector that summarizes the sequences. 

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b= np.where(a < 5, a, 10*a)
print(b)
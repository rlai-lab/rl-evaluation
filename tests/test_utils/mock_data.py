import numpy as np
import pandas as pd

from itertools import product

def generate_wide_data():
    df = pd.DataFrame(columns=['stepsize', 'optimizer', 'results'])

    alphas = [0.1, 0.01, 0.001]
    opts = ['adam', 'rmsprop']
    for i, (a, opt) in enumerate(product(alphas, opts)):
        r = np.random.randn(10, 300)
        df.loc[i] = [a, opt, r]

    return df

def generate_split_over_seed():
    df = pd.DataFrame(columns=['stepsize', 'optimizer', 'run', 'results'])

    alphas = [0.1, 0.01, 0.001]
    opts = ['adam', 'rmsprop']
    runs = range(10)
    for i, (a, opt, r) in enumerate(product(alphas, opts, runs)):
        res = np.random.randn(300)
        df.loc[i] = [a, opt, r, res]

    return df

from typing import Any, Dict
import numpy as np
import pandas as pd
import polars as pl

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


def learning_curve_data():
    hypers = {
        'alpha': [0.1, 0.01, 0.001],
        'beta': [1, 2, 4],
        'gamma': [0.99, 0.95, 0.9],
    }

    max_steps = 1000
    samples = 25
    seeds = 10

    rng = np.random.default_rng(0)

    raw_data = []
    for hyper_combo in dict_product(**hypers):
        mean_metric = rng.uniform(-10, 10)
        for s in range(seeds):
            time = rng.permutation(max_steps)[:samples]
            time.sort()
            metric = rng.normal(mean_metric, 1.0, size=samples)

            for t in range(samples):
                raw_data.append({
                    **hyper_combo,
                    'time': time[t],
                    'metric': metric[t] + t,
                    'seed': s,
                })

    return pl.DataFrame(raw_data)

def dict_product(**kwargs: Dict[str, Any]):
    keys = kwargs.keys()
    for val in product(*kwargs.values()):
        yield dict(zip(keys, val, strict=True))

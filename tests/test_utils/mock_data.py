import numpy as np
import pandas as pd

def generate_wide_data():
    df = pd.DataFrame(columns=['stepsize', 'optimizer', 'results'])

    for a in [0.1, 0.01, 0.001]:
        for opt in ['adam', 'rmsprop']:
            r = np.random.randn(300)
            df.add(([a], [opt], [r]), axis=0)

    print(df)

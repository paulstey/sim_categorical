import numpy as np
import pandas as pd

from collections import Counter


a = np.random.choice(['a', 'b', 'c'], 10)
b = np.random.choice(['a', 'b'], 10)




def gen_categorical(xtabs_df, x1):
    n = len(x1)
    x1_vals = xtabs_df.index.values
    x2 = np.zeros(10, dtype='<U256')

    xtabs_counts = xtabs_df * n         # convert proportions to counts
    used_indcs = np.array([], int)
    
    for x2_val in xtabs_counts.columns:
        for x1_val in xtabs_counts.index:
            print(x2_val)
            print(x1_val)

            candidate_indcs = [x for x in np.where(x1 == x1_val)[0] if x not in used_indcs]

            print(candidate_indcs)

            num_entries = round(xtabs_counts[x2_val][x1_val])
            print(num_entries)
            indcs = np.random.choice(candidate_indcs, num_entries, replace = False)
            x2[indcs] = val
            used_indcs = np.append(used_indcs, indcs)

    return x2


xt = pd.DataFrame([[0.40, 0.10],[0.10, 0.15], [0.20, 0.05]])
xt.columns = ['a', 'b']
xt.index = ['a', 'b', 'c']



gen_categorical(xt, a)

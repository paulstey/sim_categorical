
import numpy as np
import pandas as pd 



def verify_length(x1, x2, xtab_df, n):
    if len(x1) > n:
        x1.pop()
        x2.pop()
    elif len(x1) < n:
        x1.append(np.random.choice(xtab_df.index))
        x2.append(np.random.choice(xtab_df.columns))
    return x1, x2


def xtabs2vars(xtab_df, n):
    '''
    Given a dataframe representing the cross-tabs of two variables, 
    this function returns the two variables with values reflecting
    the proportions in the xtabs.
    '''
    x1 = []
    x2 = []
    
    for k1 in xtab_df.index:
        for k2 in xtab_df.columns:
            # get count from proportion in xtabs
            cnt = int(np.round(n * xtab_df.loc[k1, k2]))
            for i in range(cnt):
                x1.append(k1)
                x2.append(k2)

    x1, x2 = verify_length(x1, x2, xtab_df, n)

    x1 = np.array(x1)
    x2 = np.array(x2)
    
    return (x1, x2)


df = pd.DataFrame()
df['a'] = [0.3, 0.2, 0.1]
df['b'] = [0.15, 0.05, 0.2]
df.index = ['x', 'y', 'z']

x1, x2 = xtabs2vars(df, 100)
df_res = pd.DataFrame([x1, x2]).transpose()

def test_xtabs2vars(xtab_df, n_iter, max_n):
    '''
    This is for unit testing the xtabs2vars(); confirming it returns vectors 
    with correct lengths. Going from proportion to count left room for rounding error.
    '''
    for _ in range(n_iter):
        n = np.random.choice(range(max_n))
        x1, x2 = xtabs2vars(xtab_df, n)
        if len(x1) != n:
            raise Exception("Length of x1 not equal to specified length")

    return "ok"

test_xtabs2vars(df, 100, 100)



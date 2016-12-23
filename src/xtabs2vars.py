
import numpy as np

def confirm_length(x, vals, n):
    m = len(x)
    if m == n:
        res = x
    elif m > n:
        res = x[0:-2]
    elif m < n: 
        res = 

def xtabs2vars(tab_df, n):
    '''
    Given a dataframe representing the cross-tabs of two variables, 
    this function returns the two variables with their values reflecting
    those of the xtabs.
    '''
    x1 = [] #np.zeros(n, str)
    x2 = [] #np.zeros(n, str)

    k1, k2 = tab_df.shape          # num categories in 
    for k1 in tab_df.index:
        for k2 in tab_df.columns:
            print(k1, ' ', k2)
            cnt = int(np.round(n * tab_df.loc[k1, k2]))
            for i in range(cnt):
                x1.append(k1)
                x2.append(k2)
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    return (x1, x2)






df = pd.DataFrame()
df['a'] = [0.3, 0.2, 0.1]
df['b'] = [0.15, 0.05, 0.2]
df.index = ['x', 'y', 'z']
xtabs2vars(df, 10)



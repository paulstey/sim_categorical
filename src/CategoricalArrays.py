import numpy as np

class CategoricalArray:
    '''
    The class CategoricalArray is used for taking one-dimensional array-type
    (e.g., list, or np.array) objects of string data and converting to a matrix of
    dummy codes such that the resulting CategoricalArray can be used in model equations.

    x = CategoricalArray(["dog", "cat", "cat", "bird", "fish", "dog"])
    x * [4.2, 0.1, 8.0]
    '''
    def __init__(self, v):
        self.data = v
        self.categories = np.unique(v)             # NOTE: returns sorted values
        self.num_categories = len(self.categories)
        self.n = len(v)

        self.dummycodes = np.zeros((self.n, self.num_categories-1), int)
        for i in range(self.n):
            for j in range(1, self.num_categories-1):
                if self.data[i] == self.categories[j]:
                    self.dummycodes[i, j] = 1
                    break

    def __len__(self):
        return len(self.n)

    def __mul__(self, betas):
        if len(betas) != self.num_categories-1:
            raise TypeError("CategoricalArray must be multiplied by vectors of length \
                             equal to one minus the number of categories.")
        else:
            out = np.dot(self.dummycodes, np.array(betas))
        return out

    def __rmul__(self, betas):
         if len(betas) != self.num_categories-1:
             raise TypeError("CategoricalArray must be multiplied by vectors of length \
                              equal to one minus the number of categories.")
         else:
             out = np.dot(np.array(betas), self.dummycodes)
         return out

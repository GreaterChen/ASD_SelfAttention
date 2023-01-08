import random

import numpy as np
from sklearn.decomposition import PCA

X = np.array([[random.randint(1, 100) for i in range(int(116 * 115 / 2))] for j in range(115)])
print(X)
print(X.shape)
pca = PCA(n_components=1155, svd_solver='randomized')
X_r = pca.fit(X).transform(X)
print(X_r)
print(X_r.shape)


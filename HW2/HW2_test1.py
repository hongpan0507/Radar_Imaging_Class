import numpy as np

a = np.array([1, 2, 3])
for i in range(0, 5):
    t = np.array([a])
    if i == 0:
        c = t
    else:
        c = np.concatenate((c, t))
c[0, :] = [4, 5, 6]
print(c[0, :])
print(c.shape[0])
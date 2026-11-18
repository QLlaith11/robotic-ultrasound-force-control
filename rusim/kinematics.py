import numpy as np

LINKS = np.array([0.34, 0.30, 0.28, 0.22, 0.18, 0.14, 0.10])

def fk(q):
    q = np.asarray(q)
    a = np.cumsum(q)
    x = np.sum(LINKS * np.cos(a))
    y = np.sum(LINKS * np.sin(a))
    z = 0.72 + 0.03 * np.sin(q[0]) + 0.02 * np.sin(q[2])
    return np.array([x, y, z])

def jacobian(q):
    q = np.asarray(q)
    a = np.cumsum(q)
    J = np.zeros((3, 7))
    for j in range(7):
        J[0, j] = -np.sum(LINKS[j:] * np.sin(a[j:]))
        J[1, j] = np.sum(LINKS[j:] * np.cos(a[j:]))
    J[2, 0] = .03*np.cos(q[0]); J[2, 2] = .02*np.cos(q[2])
    return J

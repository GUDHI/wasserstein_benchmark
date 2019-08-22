'''
script to compute wasserstein distance between persistence diagrams using optimal transport and its regularized version
@author: Theo Lacombe
'''

import numpy as np
import scipy.spatial.distance as sc
try:
    import ot
except ImportError:
    print("POT (Python Optimal Transport) package is not installed. Try to run $ pip install POT")


def proj_on_diag(X):
    '''
    param X: a (n x 2) array encoding the points of a persistent diagram.
    return: a (n x 2) arary encoding the (respective orthogonal) projections of the points onto the diagonal
    '''
    Z = (X[:,0] + X[:,1]) / 2.
    return np.array([Z , Z]).T


def build_dist_matrix(X,Y,p=2.):
    '''
    param X: (n x 2) np.array encoding the (points of the) first diagram.
    param Y: (m x 2) np.array encoding the second diagram.
    param p: exponent for *both* the ground metric and the Wasserstein metric. That is, we compute the p-Wasserstein distance (1 <= p < infty) with respect to the p-norm as ground metric.
    return: (n+1) x (m+1) np.array encoding the cost matrix C. 
                For 1 <= i <= n, 1 <= j <= m, C[i,j] encodes the distance between X[i] and Y[j], while C[i, m+1] (resp. C[n+1, j]) encodes the distance (to the p) between X[i] (resp Y[j]) and its orthogonal proj onto the diagonal.
                note also that C[n+1, m+1] = 0  (it costs nothing to move from the diagonal to the diagonal).
    '''
    C = sc.cdist(X,Y, metric='minkowski', p=p)**p
    Xdiag = proj_on_diag(X)
    Ydiag = proj_on_diag(Y)
    Cxd = np.linalg.norm(X - Xdiag, ord=p, axis=1)**p
    Cf = np.hstack((C, Cxd[:,None]))
    Cdy = np.linalg.norm(Y - Ydiag, ord=p, axis=1)**p
    Cdy = np.append(Cdy, 0)
    Cf = np.vstack((Cf, Cdy[None,:]))
    return Cf


def Wdist(X, Y, reg=0., p=2.):
    '''
    param X, Y: (n x 2) and (m x 2) numpy array (points of persistence diagrams)
    param reg: regularization parameters for entropic smoothing. If 0., exact computation.
    param p: exponent for Wasserstein;
    return: float, estimation of the Wasserstein distance between two diagrams (exact if reg = 0.).
    '''
    M = build_dist_matrix(X,Y,p=p)
    n = len(X)
    m = len(Y)
    a = 1.0 / (n + m) * np.ones(n)  # weight vector of the input diagram. Uniform here.
    hat_a = np.append(a, m/(n+m))  # so that we have a probability measure, required by POT
    b = 1.0 / (n + m) * np.ones(m)  # weight vector of the input diagram. Uniform here.
    hat_b = np.append(b, n/(m+n))  # so that we have a probability measure, required by POT
    if reg > 0:
        ot_cost = (n+m) * ot.bregman.sinkhorn2(hat_a, hat_b, M, reg=reg)
    else:
        ot_cost = (n+m) * ot.emd2(hat_a, hat_b, M)
    return np.power(ot_cost, 1./p)


if __name__=="__main__":
    '''
    Short test script
    '''
    X = np.array([[2.7, 3.7],[9.6, 14.],[34.2, 34.974]])
    Y = np.array([[2.8, 4.45],[9.5, 14.1]])
    p = 2.
    reg = 0.
    print("Estimation of Wasserstein distance:")
    print(Wdist(X, Y, reg=reg, p=p))

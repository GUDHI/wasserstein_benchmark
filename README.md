Python3 Implementation of Wasserstein distance between persistence diagrams using optimal transport.

Dependencies:
- numpy
- scipy
- Python Optimal Transport  (can be installed with `$ pip install pot`).

Remark on entropic smoothing: 
The function `wdist` also provides an approximation of the p-Wasserstein distance between persistence diagrams using (entropic) regularized optimal transport. This regularization depends on a parameters $\gamma$ (when $\gamma \to 0$, we recover exact computation). A large $\gamma$ makes the estimation faster (but less accurate). A small $\gamma$ gives a better approximation. Note however that numerical instability can arise if $\gamma$ is too small. Namely, one must have that the entries of $C / $\gamma$) (where $C$ is the pairwise distances matrix between points) is less than 1000.

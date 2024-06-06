# DeepSplines
This repository provides a learnable linear spline module (PyTorch).

Features
------------------
* Efficient B-spline implementation of the linear spline
* Projector for imposing constraints on the slope of the spline
* Scaling parameter to automatically adjust the range of the spline
* Second-order total-variation regularization to promote linear splines with fewer regions or knots

Related Publications
------------------
1. [Learning Activation Functions in Deep (Spline) Neural Networks](https://ieeexplore.ieee.org/abstract/document/9264754/)  <br />
IEEE Open Journal of Signal Processing,vol. 1, pp. 295–309, 2020. <br />
P. Bohra, J. Campos, H. Gupta, S. Aziznejad, and M. Unser.

2. [Improving Lipschitz-Constrained Neural Networks by Learning Activation Functions](https://www.jmlr.org/papers/volume25/22-1347/22-1347.pdf)  <br />
Journal of Machine Learning Research, vol. 25, no. 65, pp. 1–30, 2024. <br />
S. Ducotterd, A. Goujon, P. Bohra, D. Perdios, S. Neumayer, and M. Unser.

Developers
------------------
This framework was developed at the Biomedical Imaging Group, École polytechnique fédérale de Lausanne (EPFL), Switzerland. This work was supported in part by the Swiss National Science Foundation under Grant 200020_184646 / 1 and in part by the European Research Council (ERC) under Grant 692726-GlobalBioIm and Grant 101020573 (Project FunLearn).

Contributors: Alexis Goujon, Joaquim Campos, Pakshal Bohra, Stanislas Ducotterd

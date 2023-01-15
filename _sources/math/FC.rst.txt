Fully Connected
^^^^^^^^^^^^^^^

This section intends to detail the equations ruling a fully connected (dense)
layer (see :class:`bgd.layers.fc.FullyConnected`).

A fully connected layer is characterized by the number of input neurons (:math:`n_{\text{in}}`)
and the number of output neurons (:math:`n_{\text{out}}`). Each of the output neurons is a linear
combination of all the input neurons with possibly a bias (intercept).

If layer :math:`k` is a fully connected one, the layer parameters are:

.. math::

   \theta^{(k)} = (A^{(k)}, b^{(k)}) \in \mathbb R^{n^{(k-1)} \times n^{(k)}} \times \mathbb R^{n^{(k)}},

and the layer function is given by:

.. math::

   \Lambda_{(A^{(k)}, b^{(k)})}^{(k)} : \mathbb R^{n^{(k-1)}} \to \mathbb R^{n^{(k)}} : x \mapsto xA^{(k)} + b^{(k)}.

Backpropagation
"""""""""""""""

The backpropagation algorithm requires :math:`\partial_{A^{(k)}_{i,j}}\mathcal L`,
:math:`\partial_{b^{(k)}_i}\mathcal L` and :math:`\partial_{X^{(k-1)}_{i,j}}\mathcal L`

For the weights update:

.. math::

   \partial_{A^{(k)}_{i,j}}\mathcal L(X) &= \sum_{\alpha,\beta = (1,1)}^{(\ell, n^{(k)})}\varepsilon^{(k)}_{\alpha,\beta}\partial_{A^{(k)}_{i,j}}X^{(k)}_{\alpha,\beta}
   = \sum_{\alpha,\beta = (1,1)}^{(\ell, n^{(k)})}\varepsilon^{(k)}_{\alpha,\beta}\sum_{\gamma=1}^{n^{(k-1)}}X_{\alpha,\gamma}\partial_{A^{(k)}_{i,j}}A^{(k)}_{\gamma,\beta} \\
                                         &= \sum_{\alpha,\beta = (1,1)}^{(\ell, n^{(k)})}\varepsilon^{(k)}_{\alpha,\beta}X_{\alpha,i}\delta_\beta^j
   = \sum_{\alpha=1}^\ell\varepsilon^{(k)}_{\alpha,j}X_{\alpha,i} = \Big(X'\varepsilon^{(k)}\Big)_{i,j}.

Therefore :math:`\nabla_{A^{(k)}}\mathcal L = X'\varepsilon`.

.. math::

   \partial_{b^{(k)}_i}\mathcal L = \sum_{\alpha,\beta = (1,1)}^{(\ell, n^{(k)})}\varepsilon^{(k)}_{\alpha,\beta}\partial_{b^{(k)}_i}b^{(k)}_\beta
   = \sum_{\alpha=1}^\ell\varepsilon^{(k)}_{\alpha,i}.

For the signal propagation:

.. math::

   \partial_{X^{(k-1)}_{i,j}}\mathcal L &= \sum_{\alpha,\beta = (1,1)}^{(\ell, n^{(k)})}\varepsilon^{(k)}_{\alpha,\beta}\partial_{X^{(k-1)}_{i,j}}X^{(k)}_{\alpha,\beta}
   = \sum_{\alpha,\beta = (1,1)}^{(\ell,n^{(k)})}\varepsilon^{(k)}_{\alpha,\beta}\sum_{\gamma=1}^{n^{(k-1)}}\delta_i^\alpha\delta_j^\gamma \\
   &= \sum_{\beta=1}^{n^{(k)}}\varepsilon^{(k)}_{i,\beta}A^{(k)}_{\beta,j} = \Big(\varepsilon^{(k)}{A^{(k)}}'\Big)_{i,j}.

Therefore :math:`\nabla_{X^{(k-1)}}\mathcal L = \varepsilon^{(k)}{A^{(k)}}'`.

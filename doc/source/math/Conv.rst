Convolutional Layer
^^^^^^^^^^^^^^^^^^^

This section intends to detail the equations ruling a (2D) convolutional layer
(see :class:`bgd.layers.conv2d.Convolutional2D`).

If layer :math:`k` is a 2D-convolutional layer (Conv2D layer), the layer
parameters are the biases (intercepts) :math:`b^{(k)}`, a 1D real vector,
and the filters :math:`\omega^{(k)}`, a 4D tensor of dimension
:math:`(n_F^{(k)}, F_H^{(k)}, F_W^{(k)}, n_C^{(k)})`, respectively the number
of filters, the width of the filters, the height of the filters, and the number
of channels in the input.

Strides
"""""""

The strides, denoted by :math:`\sigma = [\sigma_1 \; \sigma_2]' \in \mathbb {\mathbb N^*}^2`,
represent the step taken between two pixels of the output image.

They induce the following dimension for the output image:

.. math::

   (\ell, H, W, n_C^{(k)}) * (n_F^{(k)}, F_H^{(k)}, F_W^{(k)}, n_C^{(k)}) \mapsto
    (\ell, \lfloor(H - F_H^{(k)}) / \sigma_1\rfloor + 1, \lfloor(W - F_W^{(k)}) / \sigma_2\rfloor + 1, n_F^{(k)}).

Dilations
"""""""""

The dilations, denoted by :math:`\delta = [\delta_1 \; \delta_2]' \in \mathbb {\mathbb N^*}^2`,
represent a dilation on the filters, i.e. the number of rows/columns of pixels in the input
image that are skipped between two rows/columns of the filters.

The effect of the dilations can be viewed as a *literal* dilation of the filters where a
2D filter :math:`\omega` of shape :math:`(m, n)` is dilated into another 2D filter
:math:`\hat \omega` of shape :math:`(\hat m, \hat n) = (\delta_1 \cdot (m-1) + 1, \delta_2 \cdot (n-1) + 1)`.

Therefore, by using the notation :math:`\hat .` in order to denote the shape of a
dilated filter, we have that the dimension of the output image is:

.. math::

   (\ell, \lfloor(H - \hat F_H^{(k)}) / \sigma_1\rfloor + 1, \lfloor(W - \hat F_W^{(k)}) / \sigma_2\rfloor + 1, n_F^{(k)}),

where :math:`\hat F_H^{(k)} = 1 + \delta_1(F_H^{(k)} - 1)` and :math:`\hat F_W^{(k)} = 1 + \delta_2(F_W^{(k)} - 1)`.

Convolution function
""""""""""""""""""""

The layer function is then given by:

.. math::

	\Lambda^{(k)}_{\theta^{(k)}} :
	 \mathbb R^{\ell \times H \times W \times n_C^{(k)}}
	   \to \mathbb R^{\ell \times (\lfloor(H - \hat F_H^{(k)}) / \sigma_1\rfloor+1) \times (\lfloor(W - \hat F_W^{(k)}) / \sigma_2\rfloor+1) \times n_F^{(k)}} :
	 X \mapsto \Lambda^{(k)}_{\theta^{(k)}}(X; \sigma, \delta),

where, for :math:`\beta` a multi-index of the output image:

.. math::

   \Lambda^{(k)}_{\theta^{(k)}}(X^{(k-1)}; \sigma, \delta)_\beta = \sum_{\gamma_1=1}^{F_H^{(k)}}\sum_{\gamma_2=1}^{F_W^{(k)}}\sum_{\gamma_3=1}^{n_C^{(k)}}\omega^{(k)}_{\beta_3,\gamma_1,\gamma_2,\gamma_3}X^{(k-1)}_{\beta_0,\beta_1+\sigma_1\gamma_1,\beta_2+\sigma_2\gamma_2,\gamma_3} + b^{(k)}_{\beta_3}.

Backpropagation
"""""""""""""""

The backpropagation algorithm requires :math:`\partial_{\omega^{(k)}_\alpha}\mathcal L`,
:math:`\partial_{b^{(k)}_i}\mathcal L` and :math:`\partial_{X^{(k-1)}_\alpha}\mathcal L`.


For the weights update:

.. math::

   \partial_{\omega^{(k)_\alpha}}X^{(k)}_\beta &= \sum_{\gamma_1}\sum_{\gamma_2}\sum_{\gamma_3}\delta^{\beta_3}_{\alpha_0}\delta^{\gamma_1}_{\alpha_1}\delta^{\gamma_2}_{\alpha_2}\delta^{\gamma_3}_{\alpha_3}X^{(k-1)}_{\beta_0,\sigma_1\beta_1 + \delta_1\gamma_1, \sigma_2\beta_2 + \delta_2\gamma_2,\gamma_3} \\
     &= \delta^{\beta_3}_{\alpha_0}X^{(k-1)}_{\beta_0,\sigma_1\beta_1 + \delta_1\alpha_1,\sigma_2\beta_2 + \delta_2\alpha_2,\alpha_3}.

Therefore:

.. math::

   \partial_{\omega^{(k)}_\alpha}\mathcal L &= \sum_{\beta_0,\beta_1,\beta_2,\beta_3}\varepsilon^{(k)}_\beta\partial_{\omega^{(k)}_\alpha}X^{(k)}_\beta
   = \sum_{\beta_0,\beta_1,\beta_2}\varepsilon^{(k)}_{\beta_0,\beta_1,\beta_2,\alpha_0}X^{(k)}_{\beta_0,\sigma_1\beta_1 + \delta_1\alpha_1,\sigma_2\beta_2 + \delta_2\alpha_2,\alpha_3} \\
   &= \sum_{\beta_0,\beta_1,\beta_2}\pi_{\tau_{0,3}}\varepsilon^{(k)}_{\alpha_0,\beta_1,\beta_2,\beta_0}\pi_{\tau_{0,3}}X^{(k-1)}_{\alpha_3,\sigma_1\beta_1 + \delta_1\alpha_1,\sigma_2\beta_2 + \delta_2\alpha_2,\beta_0}
   = \pi_{\tau_{0,3}}\Lambda^{(k)}_{(\pi_{\tau_{0,3}}\varepsilon^{(k)}, \mathbf{0})}(\pi_{\tau_{0,3}}X^{(k-1)}; \delta, \sigma)_\alpha,

i.e. the backward pass is also a convolution where strides and dilations have been swapped and without biases.

For the biases, it is more trivial:

.. math::

   \partial_{b^{(k)}_i}\mathcal L = \sum_{\beta}\varepsilon^{(k)}_\beta\partial_{b^{(k)}_i}X^{(k)}_\beta
   = \sum_{\beta}\varepsilon^{(k)}_\beta\delta_i^{\beta_3} = \sum_{\beta_0,\beta_1,\beta_2}\varepsilon^{(k)}_{\beta_0,\beta_1,\beta_2,i}.

Finally, for the signal propagation, the formula is way less trivial:

.. math::

   \partial_{X^{(k-1)}_\alpha}\mathcal L = \sum_{\beta}\varepsilon^{(k)}_\beta\partial_{X^{(k-1)}_\alpha}X^{(k)}_\beta
    = \sum_{\beta_1}\sum_{\beta_2}\sum_{\beta_3}\sum_{\gamma_1}\sum_{\gamma_2}\varepsilon^{(k)}_{\alpha_0,\beta_1,\beta_2,\beta_3}\omega^{(k)}_{\beta_3,\gamma_1,\gamma_2,\alpha_3}\delta_{\sigma_1\beta_1+\delta_1\gamma_1}^{\alpha_1}\delta_{\sigma_2\beta_2+\delta_2\gamma_2}^{\alpha_2}.

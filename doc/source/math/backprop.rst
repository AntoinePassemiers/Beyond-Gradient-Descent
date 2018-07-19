Backpropagation
---------------

Backpropagation is an iterative process aiming to find the optimum of a cost (loss) function.

Each parameter :math:`\theta^{(k)}_\alpha` is iteratively updated with the following formula:

.. math::

   \theta^{(k)}_\alpha \leftarrow \theta^{(k)}_\alpha - \eta\partial_{\theta^{(k)}_\alpha}\mathcal L(X).

This partial derivative can be computed using the chain rule:

.. math::

   \partial_{\theta^{(k)}_\alpha}\mathcal L = \sum_\beta\partial_{X^{(k)}_\beta}\mathcal L\partial_{\theta^{(k)}_\alpha}X^{(k)}_\beta.

For the sake of convenience, we write :math:`\varepsilon^{(k)} = \nabla_{X^{(k)}}\mathcal L`, which is
the signal back-propagated by layer :math:`k` to layer :math:`k-1`.

Therefore, each layer needs to compute the derivative of the loss w.r.t. its parameters in order to move
towards the optimum, but each layer also needs to compute the derivative of the loss w.r.t. its input, i.e.
:math:`\nabla_{X^{(k)}}`.

That's where the name *back propagation* comes from: each layer needs to propagate a signal to the layer before
it so that all the derivatives w.r.t. parameters can be computed properly.

The backpropagation starts by taking the derivative of the loss w.r.t. the predictions:
:math:`\nabla_{\hat y}\mathcal L = \nabla_{X^{(K)}}\mathcal L = \varepsilon^{(K)}`.
This is why the cost functions defined in :mod:`bgd.cost` need to have a :obj:`_grad` method defined:
differentiation is automatized.

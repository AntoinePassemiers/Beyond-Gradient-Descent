Introduction and Notations
--------------------------

Dataset
"""""""

Starting from here, we assume that :math:`\mathcal D = \{(x^{(i)}, y^{(i)}\}_{i=1}^N`
is the dataset such that:

.. math::

   \forall i \in \{1, \ldots, N\} : x^{(i)} \in \mathbb R^n \text{ and } y^{(i)} \in \mathbb R^m.

Let :math:`\mathcal N` denote the neural net function:

.. math::

   \mathcal N : \mathbb R^n \to \mathbb R^m : x \mapsto \mathcal N(x).

For :math:`X \in \mathbb R^{\ell \times n}` a matrix of samples (each row is a
sample in :math:`\mathcal D`), we naturally extend :math:`\mathcal N` such that:

.. math::

   \mathcal N(X) = [\mathcal N(X_i)]_{i=1}^\ell \in \mathbb R^{\ell \times m}.

Neural Stack
""""""""""""

A linear neural network (LNN for short, also called neural stack, see
:class:`bgd.nn.NeuralStack`) is a composition of a list of layers. Let's denote
the number of layers by :math:`K \in \mathbb N^*`. For :math:`k \in \{1, \ldots, K\}`,
the :math:`k` th layer is denoted by :math:`\Lambda^{(k)}` such that:

.. math::

   \Lambda^{(k)}_{\theta^{(k)}} : \mathbb R^{n^{(k-1)}} \to \mathbb R^{n^{(k)}},

where :math:`\theta^{(k)}` denotes the parameters of layer :math:`k` and is a
tensor or rank :math:`r^{(k)} \in \mathbb N` (0 if non-parametrized layer) and
dimension :math:`\delta^{(k)} \in \mathbb {N^*}^{r^{(k)}}` such that:

.. math::

   \theta^{(k)} \in \mathbb R^{\delta^{(k)}} := \mathbb R^{\prod_{i=1}^{r^{(k)}}\delta^{(k)}_i},

:math:`n^{(j)}` denotes the dimension of the (tensor) output of the :math:`j` th layer.
For sake of notation, we also introduce :math:`n^{(0)} = n`.

The neural net is therefore defined as the composition of all the layers:

.. math::

   \mathcal N(X) := \Big(\bigcirc_{k=1}^K\Lambda_{\theta^{(k)}}^{(k)}\Big)(X),

where the composition must be read :math:`\Lambda^{(K)}_{\theta^{(k)}} \circ \ldots \circ \Lambda^{(1)}_{\theta^{(1)}}`
so that dimensions stay consistent.

Miscelanous Notations
"""""""""""""""""""""

We also write :math:`X \in \mathbb R^{\ell \times n}` for a batch input of the LNN.

For sake of convenience, for :math:`k \in \{1, \ldots, K\}`, we write:
:math:`X^{(k)} := \Lambda_{\theta^{(k)}}^{(k)}(X^{(k-1)})` and :math:`X^{(0)} := X`.

For :math:`X` a rank :math:`n` tensor and :math:`\sigma \in \mathfrak S_n` a permutation on
:math:`n` elements, we introduce the following notation:

.. math::

   \pi_\sigma X := \pi_\sigma(X) := [X_{\sigma\alpha}]_\alpha,

an extension of the transposition of matrices: if :math:`\sigma = \tau_{i,j}`, then
:math:`\pi_\sigma X` is still a rank :math:`n` tensor but whose indices :math:`i`
and :math:`j` have been swapped.

Cost Function
"""""""""""""

A cost function (or *loss function* which is equivalent) is a function:

.. math::

   L : \mathbb R^{m} \times \mathbb R^m \to \mathbb R^+ : (y, \hat y) \mapsto L(y, \hat y)

that represents the error between ground truth :math:`y` and estimation :math:`\hat y = \mathcal N(x)`.
The training phase will attempt to reach the minimum of the loss :math:`L` by adapting the
parameters of each layer.

We introduce the notation:

.. math::

   \mathcal L : \mathbb R^{\ell \times n} \to \mathbb R^+ : X \mapsto L(\mathbf y, \mathcal N(X))

for the loss computed on samples of the dataset :math:`\mathcal D` where :math:`\mathbf y`
is the label vector associated to samples :math:`X` (i.e. :math:`\mathbf y = [y^{(i_j)}]_{j=1}^\ell`
for :math:`i_j \in \{1, \ldots, N\}` such that :math:`X = [x^{(i_j)}]_{j=1}^\ell`).

The list of loss functions implemented in BGD can be found in the :mod:`bgd.cost`.

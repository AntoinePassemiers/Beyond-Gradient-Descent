Introduction and Notations
--------------------------

Starting from here, we assume that :math:`\mathcal D = \{(x^{(i)}, y^{(i)}\}_{i=1}^N`
is the dataset such that:

.. math::

   \forall i \in \{1, \ldots, N\} : x^{(i)} \in \mathbb R^n \text{ and } y^{(i)} \in \mathbb R^m.

Let :math:`\mathcal N` denote the neural net:

.. math::

   \mathcal N : \mathbb R^n \to \mathbb R^m : x \mapsto \mathcal N(x).

For :math:`X \in \mathbb R^{\ell \times n}` a matrix of samples (each row is a
sample in :math:`\mathcal D`), we naturally extend :math:`\mathcal N` such that:

.. math::

   \mathcal N(X) = [\mathcal N(X_i)]_{i=1}^k \in \mathbb R^{\ell \times n}.

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

:math:`n^{(j)}` denotes the dimension of the output of the :math:`j` th layer.
For sake of notation, we also introduce :math:`n^{(0)} = n`.

The neural net is therefore defined as the composition of all the layers:

.. math::

   \mathcal N(X) := \Big(\bigcirc_{k=1}^K\Lambda_{\theta^{(k)}}^{(k)}\Big)(X),

where the composition must be read :math:`\Lambda^{(K)}_{\theta^{(k)}} \circ \ldots \circ \Lambda^{(1)}_{\theta^{(1)}}`
so that dimensions stay consistent.

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

MOT DRO 
===========

Transport configuration
-----------------------

Configure the ambiguity radius, feature norm, and transport exponent with
``model.update({...})``:

* ``eps``: non-negative OT ambiguity radius.
* ``p``: feature perturbation norm order, one of ``1``, ``2``, or ``'inf'``
  (default: ``2``).
* ``square``: transport exponent, either ``1`` or ``2`` (default: ``2``).

Targets remain fixed, and the feature transport cost is
``||x - x_hat||_p ** square``. For example, ``p=2, square=2`` selects squared
Euclidean transport; ``p=2, square=1`` selects Euclidean transport.
Unsupported norms or exponents raise ``MOTDROError``.

.. code-block:: python

   from dro.linear_model.mot_dro import MOTDRO

   model = MOTDRO(input_dim=2, model_type='lad')
   model.update({'eps': 0.1, 'p': 2, 'square': 2})

.. automodule:: dro.linear_model.mot_dro
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: MOTDROError

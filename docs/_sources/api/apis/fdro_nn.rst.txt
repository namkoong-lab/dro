f-Divergence DRO 
==================

Training customization
----------------------

``Chi2NNDRO`` and ``CVaRNNDRO`` inherit the :ref:`neural training controls
<neural-fit-customization>`. Their ``fit(...)`` methods accept paired
``train_loader`` and ``val_loader`` arguments, a PyTorch ``optimizer``, and
an optional ``scheduler`` with ``scheduler_step="epoch"`` or ``"batch"``.
Build the optimizer after any ``update_user_mode(...)`` call so that it
updates the current network. The existing ``fit(X, y, ...)`` path remains
available with an internal split and Adam optimizer.

Radius certificates
-------------------

``Chi2NNDRO.certify_radius(losses, candidate_weights, eta, epsilon=None)``
returns ``epsilon_cert``. ``CVaRNNDRO.certify_radius(losses,
candidate_weights, eta, alpha=None)`` returns ``alpha_cert``; a larger tail
fraction means a smaller ambiguity set. Both methods also return the
certified empirical loss bound, the upper bound at the input parameter, and
their gap, or ``None`` when the reduced parameter cannot be certified at the
candidate loss. Pass full-data per-example losses and feasible weights for
the final frozen network. See the :doc:`certificate derivations and runnable
example <../../tutorials/neural_certificates>`.

.. automodule:: dro.neural_model.fdro_nn
   :members:
   :undoc-members:
   :show-inheritance:

Sinkhorn DRO 
=================

Training customization
----------------------

``SinkhornNNDRO`` also inherits the :ref:`neural training controls
<neural-fit-customization>`. Supply both ``train_loader`` and ``val_loader``
to use external batches, and pass an ``optimizer`` and optional ``scheduler``
to control parameter updates and the learning-rate schedule. Construct the
optimizer after any ``update_user_mode(...)`` call. The original
``fit(X, y, ...)`` path remains available.


.. automodule:: dro.neural_model.sinkhorn_nn
   :members:
   :undoc-members:
   :show-inheritance:

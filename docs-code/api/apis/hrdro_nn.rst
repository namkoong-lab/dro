Holistic Robust DRO 
=====================

Training customization
----------------------

``HRNNDRO.fit(...)`` accepts paired ``train_loader`` and ``val_loader``
arguments, a PyTorch ``optimizer``, and an optional ``scheduler``. Select
``scheduler_step="epoch"`` (default) or ``"batch"``. Build the optimizer
after calling ``update_user_mode(...)``; that call also rebinds the
adversarial attack to the replacement network. The loader batch size can
differ from ``train_batch_size``: HR-DRO rebuilds its weighting problem
when the size changes. See the :ref:`shared training rules
<neural-fit-customization>` for validation and defaults.

.. automodule:: dro.neural_model.hrdro_nn
   :members:
   :undoc-members:
   :show-inheritance:

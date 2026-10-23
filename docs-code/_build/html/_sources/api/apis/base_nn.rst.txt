NN DRO Base Class 
====================

.. _neural-fit-customization:

Training controls
-----------------

Neural DRO models using ``BaseNNDRO.fit`` accept either arrays through
``fit(X, y, ...)`` or a caller-provided pair of ``train_loader`` and
``val_loader``. Each loader must yield ``(inputs, labels)`` tensor batches.
Do not pass ``X`` or ``y`` with external loaders. The array path keeps the
internal training/validation split and Adam optimizer.

Pass ``optimizer`` to use a different PyTorch optimizer. Build it from the
current ``model.model.parameters()`` after any ``update_user_mode(...)`` call.
A ``scheduler`` requires that explicit optimizer. Set ``scheduler_step`` to
``"epoch"`` (the default) or ``"batch"``. ``ReduceLROnPlateau`` supports only
epoch stepping and receives the mean training loss for that epoch.

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader, TensorDataset
   from dro.neural_model.fdro_nn import Chi2NNDRO

   X = torch.rand(24, 2)
   y = (X[:, 0] > 0.5).long()
   train_loader = DataLoader(TensorDataset(X[:20], y[:20]), batch_size=4)
   val_loader = DataLoader(TensorDataset(X[20:], y[20:]), batch_size=4)

   model = Chi2NNDRO(input_dim=2, num_classes=2, model_type="linear")
   optimizer = torch.optim.SGD(model.model.parameters(), lr=0.01)
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
   metrics = model.fit(
       train_loader=train_loader,
       val_loader=val_loader,
       optimizer=optimizer,
       scheduler=scheduler,
       scheduler_step="epoch",
       epochs=2,
       verbose=False,
   )

See the :doc:`personalization tutorial <../../tutorials/personal_loss>` for
custom networks, losses, and examples for the individual DRO methods.

.. automodule:: dro.neural_model.base_nn
   :members:
   :exclude-members: DROError, ParameterError, DataValidationError
   :show-inheritance:

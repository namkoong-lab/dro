Wasserstein DRO 
=================

Training customization
----------------------

``WNNDRO`` inherits the :ref:`neural training controls <neural-fit-customization>`.
Its ``fit(...)`` method accepts paired ``train_loader`` and ``val_loader``
arguments, a PyTorch ``optimizer``, and an optional ``scheduler`` with
``scheduler_step="epoch"`` or ``"batch"``. Install a custom network with
``update_user_mode(...)`` before constructing the optimizer; in
``norm_ball`` mode, this also rebinds the attack to the replacement network.
These training controls do not change the attack's loss: a custom ``_loss``
affects the inner search in ``transport_penalty`` mode, but only the outer
objective in ``norm_ball`` mode.
The :doc:`WNNDRO tutorial <../notebooks/neural_dro_tutorial>` includes a
"Custom WNNDRO architecture and training controls" example with a
user-defined PyTorch ``nn.Module``, external loaders, an optimizer, and a
scheduler. The custom module must accept the selected input batches and return
task-compatible outputs (class logits for this classification example).

Radius certificate
------------------

``WNNDRO.certify_radius(transport_budget, adversarial_losses,
transport_costs, certified_inner_errors, transport_penalty=None)`` returns
``epsilon_cert`` and its empirical loss bound when the supplied global inner
error bounds suffice. It returns ``None`` when no nonnegative budget can be
certified at that loss level. The constructor's ``epsilon`` in ``norm_ball``
mode is a per-example attack radius, not ``transport_budget``. See the
:doc:`certificate derivation and runnable example
<../../tutorials/neural_certificates>`.


.. automodule:: dro.neural_model.wdro_nn
   :members:
   :undoc-members:
   :show-inheritance:

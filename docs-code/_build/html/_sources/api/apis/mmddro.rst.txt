MMD DRO 
==========

``MMD_DRO`` fits a linear predictor while a kernel defines the MMD ambiguity
set on joint feature--target samples. The default kernel is Gaussian RBF;
Laplacian, polynomial, and linear kernels are also available.

``fit(X, y, accelerate=True)`` (the default) approximates the kernel with a
Nyström feature map fitted once on the sampled support, then transforms that
support in parallel batches. It enforces the loss constraints on at most
5,000 sampled support points. Set ``n_components`` through
``update_kernel({"n_components": ...})`` to choose the feature-map rank;
the default is the smaller of 100 and the support size. With
``accelerate=False``, fitting instead uses the full kernel Gram matrix and
constraints for every sampled support point. In the accelerated mode,
``accelerated_regularizer="feature_coefficients"`` is the default and uses
the feature-coefficient norm corresponding to the approximate RKHS penalty.
``"support_values"`` retains the legacy penalty on sampled function values,
which differs from the RKHS penalty. Nyström approximation and constraint
subsampling still distinguish accelerated fitting from the full formulation.

.. automodule:: dro.linear_model.mmd_dro
   :members:
   :undoc-members:
   :show-inheritance: 
   :exclude-members: MMDDROError

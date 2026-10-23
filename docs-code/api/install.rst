Installation Guide
====================

(1) Install ``dro`` package
-----------------------------

To install ``dro`` package, you can simply run:

.. code-block:: bash

   pip install dro

And it will install all required packages.

(2) Solver selection
--------------------

The convex linear DRO models use ``CLARABEL`` through ``cvxpy`` by default. No commercial solver license is needed. You can select another installed solver when it supports the model's formulation, either during initialization or before fitting:

.. code-block:: python

   from dro.linear_model.chi2_dro import Chi2DRO

   model = Chi2DRO(input_dim=2, solver='SCS')

Alternatively, assign ``model.solver = 'SCS'`` before calling ``fit()``.

See the `CVXPY solver guide <https://www.cvxpy.org/tutorial/solvers/index.html>`_ for solver capabilities and options.

``MOSEK`` is optional. To use it, run ``pip install mosek``, obtain a license, and pass ``solver='MOSEK'`` to the model. To prepare the license:

* Request a license from the `MOSEK website <https://www.mosek.com/products/academic-licenses/>`_.
* Put the ``mosek.lic`` file in your home directory:

  .. code-block:: bash

     cd
     mkdir mosek
     mv /path_to_license/mosek.lic mosek/

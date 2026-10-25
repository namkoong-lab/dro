.. DRO documentation master file

Welcome to DRO Package Documentation
=========================================

Documentation for ``dro`` |release|, a Python package for distributionally
robust optimization in machine learning. It supports linear and kernel models,
neural networks, and tree ensembles, along with synthetic data generation and
model evaluation tools.

The convex linear models use the open-source ``CLARABEL`` solver through
``cvxpy`` by default; no commercial solver license is required. Start with the
:doc:`installation guide <api/install>`, explore the :doc:`tutorials
<api/tutorials>`, or browse the :doc:`API reference <api/api>`.

.. toctree::
   :maxdepth: 2

   api/about
   api/install
   api/tutorials
   api/example
   api/api

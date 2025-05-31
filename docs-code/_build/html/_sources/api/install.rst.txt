Installation Guide
====================

(1) Prepare ``Mosek`` license
-------------------------------

Our package is built upon ``cvxpy`` and ``mosek`` (optional), which needs the license file. The steps are as follows.

If you want to use ``mosek`` as the solver, please:

* Request license at `Official Website <https://www.mosek.com/products/academic-licenses/>`_, and then the license ``mosek.lic`` will be emailed to you.
* Put your license in your home directory as follows:

  .. code-block:: bash

     cd
     mkdir mosek
     mv /path_to_license/mosek.lic mosek/


Otherwise, you can set the solver among some open-source solvers such as `ECOS`, `SCS` in `cvxpy` (see "https://www.cvxpy.org/tutorial/solvers/index.html" for more details). In any given DRO model, this can be done during initialization:

   .. code-block:: bash

      model = XXDRO(..., solver = 'ECOS')
    

by simply updating after initialization:

   .. code-block:: bash
      
      model.solver = 'ECOS'
   

These solvers can solve all the optimization problems implemented in the package as well.


(2) Install ``dro`` package
-----------------------------

To install ``dro`` package, you can simply run:

.. code-block:: bash

   pip install dro

And it will install all required packages.
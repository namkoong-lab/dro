Installation
===============

(1) Prepare ``Mosek`` license
-------------------------------

Our package is built upon ``cvxpy`` and ``mosek``, which needs the license file. The steps are as follows:

* Request license at `Official Website <https://www.mosek.com/products/academic-licenses/>`_, and then the license ``mosek.lic`` will be emailed to you.
* Put your license in your home directory as follows:

  .. code-block:: bash

     cd
     mkdir mosek
     mv /path_to_license/mosek.lic mosek/

(2) Install ``dro`` package
-----------------------------

To install ``dro`` package, you can simply run:

.. code-block:: bash

   pip install dro

And it will install all required packages.
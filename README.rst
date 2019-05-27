PETScOPT: A framework for high performance PDE-constrained optimization with PETSc
==================================================================================


Overview
--------

This framework provides high-level abstraction to compute gradient,
hessians and tangent linear models for PDE-constrained optimization
problems with ODE constraints.
It is based on `PETSc <http://www.mcs.anl.gov/petsc/>`_, the
*Portable, Extensible Toolkit for Scientific Computation*. The FEM
capability is based on the C++ library `MFEM <http://www.mfem.org/>`_.


Basic Installation
------------------

First `install PETSc
<http://www.mcs.anl.gov/petsc/documentation/installation.html>`_,
and then set appropriate values for ``PETSC_DIR`` and ``PETSC_ARCH`` in your
environment (PETSc version greater or equal 3.11 is required)::

  $ export PETSC_DIR=/home/user/petsc
  $ export PETSC_ARCH=arch-linux2-c-debug

Clone the `Git <http://git-scm.com/>`_ repository
hosted at `Github <https://https://github.com/stefanozampini/petscopt>`_ ::

  $ git clone https://github.com/stefanozampini/petscopt.git

Finally, enter PETScOPT top level directory and use ``make`` to compile
the code and build the PETScOPT library::

  $ cd petscopt
  $ make

Example codes are located at ``src/ts/examples/tests`` and ``src/tao/examples/tests``.

MFEM support
------------

In order to enable the MFEM based FEM layer (MFEM version greater or equal 4.0 must be used), you have two options

  **recommended**: Configure PETSc using --download-mfem --download-mfem-commit=v4.0 --download-hypre --download-metis

  **harder**: After having built PETSc, follow the instructions to `build MFEM <https://mfem.org/building/>`_, and enable PETSc as a third-party package via ``MFEM_USE_PETSC=YES``. Then set the environment variable ``MFEM_DIR``::

              $ export MFEM_DIR=_location_of_MFEM_

Then, the PETScOPT library can be built as::

  $ cd petscopt
  $ make distclean
  $ make config with_mfem=1
  $ make

Example codes are located at ``src/mfemopt/examples/tests``.


Testing
-------

In order to check the successfull build of the library type::

  $ make check

The PETScOPT library uses the PETSc testing infrastructure. To run the complete test suite type::

  $ make test


Acknowledgments
---------------

This project was supported by the Extreme Computing Research Center
Division of Computer, Electrical, and Mathematical Sciences & Engineering
(`CEMSE <http://cemse.kaust.edu.sa/>`_), King Abdullah University of
Science and Technology (`KAUST <http://www.kaust.edu.sa/>`_).

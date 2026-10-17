uv (recommended)
================

`uv`_ is a fast Python package and project manager: it installs Python itself if needed, creates virtual
environments and installs packages from `PyPI <https://pypi.org/project/pycalphad/>`_.
After `installing uv <https://docs.astral.sh/uv/getting-started/installation/>`_, create an environment called
``pycalphad-env`` and install pycalphad into it on Linux and macOS/OSX:

.. code-block:: bash

   uv venv --python 3.13 pycalphad-env
   source pycalphad-env/bin/activate
   uv pip install pycalphad

On Windows, activate the environment with ``pycalphad-env\Scripts\activate`` instead.
The environment must be activated every time you use pycalphad.
pycalphad needs Python 3.11 or later; ``--python 3.13`` makes uv create the environment with Python 3.13 (it
downloads it if necessary) rather than with whichever Python it finds first, which may be too old.

If you work in a uv project, add pycalphad as a dependency instead, and run your code with ``uv run``:

.. code-block:: bash

   uv init --python 3.13 my-project
   cd my-project
   uv add pycalphad
   uv run python my_script.py

pip
===

To install pycalphad from `PyPI <https://pypi.org/project/pycalphad/>`_ using pip:

.. code-block:: bash

   pip install -U pip setuptools
   pip install -U pycalphad

pycalphad needs Python 3.11 or later (check with ``python --version``).
A recommended best practice is to install pycalphad into a virtual environment.
To create an environment called ``pycalphad-env`` on Linux and macOS/OSX:

.. code-block:: bash

   python -m venv pycalphad-env
   source pycalphad-env/bin/activate
   pip install -U pip setuptools
   pip install -U pycalphad

This environment must be activated every time you use pycalphad.

conda
=====

`Anaconda`_ is a distribution platform for scientific Python packages created by Anaconda, Inc.
It comes bundled with the ``conda`` package manager to install Python and non-Python packages.
If you don't have Anaconda already, you can download the `Miniconda`_ distribution.

To install pycalphad from `conda-forge <https://github.com/conda-forge/pycalphad-feedstock/>`_ using conda:

.. code-block:: bash

   conda install -c conda-forge pycalphad


Updating pycalphad
==================

To update an installed pycalphad to the latest release, use the package manager it was installed with, in the
same environment:

.. code-block:: bash

   uv pip install -U pycalphad                                   # uv, in the activated environment
   uv lock --upgrade-package pycalphad && uv sync                # uv project
   pip install -U pycalphad                                      # pip
   conda update -c conda-forge pycalphad                         # conda

To check which version is installed:

.. code-block:: bash

   python -c "import pycalphad; print(pycalphad.__version__)"

To update a development version, see `Upgrading`_ below.


Development version
===================

Installing
----------

The source code for the latest development version of pycalphad is available on `GitHub <https://github.com/pycalphad/pycalphad>`_.
You will need a working version of  `git`_ to download the source code.
Installing development versions of pycalphad also requires a working C++ compiler.

* **Windows:** Install `git`_ and `Microsoft Visual C++ Build Tools version 14.X <https://visualstudio.microsoft.com/downloads/>`_.
  If you are unfamiliar with the process, you can find a `tutorial here <https://beenje.github.io/blog/posts/how-to-setup-a-windows-vm-to-build-conda-packages/#developer-tools-installation>`_.

To install an editable version of the latest development branch of pycalphad with `uv`_ (as in the
`contributor guide <https://github.com/pycalphad/pycalphad/blob/develop/ONBOARDING.rst>`_), run:

.. code-block:: bash

   git clone https://github.com/pycalphad/pycalphad.git
   cd pycalphad
   uv sync

``uv sync`` creates a virtual environment in ``.venv`` with a suitable Python, installs the dependencies and the
development tools, and builds pycalphad. Then run the automated tests to ensure everything installed correctly:

.. code-block:: bash

   uv run pytest pycalphad

Without uv, install it with pip into an environment of your choice instead:

.. code-block:: bash

   git clone https://github.com/pycalphad/pycalphad.git
   cd pycalphad
   pip install -U pip setuptools
   pip install -U --editable .
   pytest pycalphad

Upgrading
---------

Changes to Python files (``.py`` extension) in an editable install will be reflected as soon a file is saved.
Changes to `Cython`_ files (``.pyx`` and ``.pxd`` extensions) must be recompiled before they take effect.
With uv, ``uv sync`` (or any ``uv run``) recompiles them when they have changed.
Otherwise, the Cython files can be recompiled by running (from the root directory [#f1]_ of the project):

.. code-block:: bash

   python setup.py build_ext --inplace

By default, the development version installed will track the latest changes in
the ``develop`` branch of pycalphad
`on GitHub <https://github.com/pycalphad/pycalphad>`_.

To update the code in the current branch to the latest changes on GitHub, the
changes are pulled from GitHub by running ``git pull``, followed by ``uv sync`` (or the recompilation above) so
that changed Cython files and dependencies take effect.

To switch to a different branch, e.g. ``master`` (which tracks the latest
released version) or another feature branch, run ``git checkout <branch>``,
where ``<branch>`` is the name of the branch to checkout, such as ``master``
(without the brackets ``<`` and ``>``).


.. _Anaconda: https://anaconda.com/download
.. _uv: https://docs.astral.sh/uv/
.. _Cython: https://cython.org/
.. _git: https://git-scm.com/
.. _`Jupyter Notebook`: http://jupyter.readthedocs.io/en/latest/index.html
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html

.. [#f1] The "root directory" is the the top level project directory containing the ``pyproject.toml`` file, the ``pycalphad/`` package directory, etc.
.. _Installation:

Installation
============

This chapter will describe how to get, compile and run the software.

|es| releases are available as source code packages from the homepage [1]_.
This is where new users should get the code. The code within release packages
is tested and known to run on a number of platforms.
Alternatively, people who want to use the newest features of |es| or
start contributing to the software can instead obtain the
current development code via the version control system software  [2]_
from |es|'s project page at GitHub  [3]_. This code might be not as well
tested and documented as the release code; it is recommended to use this
code only if you have already gained some experience in using |es|.

Unlike most other software, no binary distributions of |es| are available,
and the software is usually not installed globally for all users.
Instead, users of |es| should compile the software themselves. The reason for
this is that it is possible to activate and deactivate various features
before compiling the code. Some of these features are not compatible
with each other, and some of the features have a profound impact on the
performance of the code. Therefore it is not possible to build a single
binary that can satisfy all needs. For performance reasons a user
should always activate only those features that are actually needed.
This means, however, that learning how to compile is a necessary evil.
The build system of |es| uses CMake to compile
software easily on a wide range of platforms.

.. _Requirements:

Requirements
------------

The following tools and libraries, including their header files,
are required to be able to compile and use |es|:

.. glossary::

    CMake
        The build system is based on CMake version 3 or later [4]_.

    C++ compiler
        The C++ core of |es| needs to be built by a C++14-capable compiler.

    Boost
        A number of advanced C++ features used by |es| are provided by Boost.
        We strongly recommend to use at least Boost 1.71.

    FFTW
        For some algorithms like P\ :math:`^3`\ M, |es| needs the FFTW library
        version 3 or later [5]_ for Fourier transforms, including header files.

    CUDA
        For some algorithms like P\ :math:`^3`\ M,
        |es| provides GPU-accelerated implementations for NVIDIA GPUs.
        We strongly recommend CUDA 12.0 or later [6]_.

    MPI
        An MPI library that implements the MPI standard version 1.2 is required
        to run simulations in parallel. |es| is currently tested against
        `Open MPI <https://www.open-mpi.org>`__ and
        `MPICH <https://www.mpich.org>`__, with and without
        `UCX <https://openucx.org>`__ enabled.
        Other MPI implementations like Intel MPI should also work, although
        they are not actively tested in |es| continuous integration.

        Open MPI version 4.x is known to not properly support the MCA binding
        policy "numa" in singleton mode on a few NUMA architectures.
        On affected systems, e.g. AMD Ryzen or AMD EPYC, Open MPI halts with
        a fatal error when setting the processor affinity in ``MPI_Init``.
        This issue can be resolved by setting the environment variable
        ``OMPI_MCA_hwloc_base_binding_policy`` to a value other than "numa",
        such as "l3cache" to bind to a NUMA shared memory block, or to
        "none" to disable binding (can cause performance loss).

    Python
        |es|'s main user interface relies on Python 3.

        We strongly recommend using Python environments to isolate
        packages required by |es| from packages installed system-wide.
        This can be achieved using venv [7]_, conda [8]_, or any similar tool.
        Inside an environment, commands of the form
        ``sudo apt install python3-numpy python3-scipy``
        can be rewritten as ``python3 -m pip install numpy scipy``,
        and thus do not require root privileges.

        Depending on your needs, you may choose to install all |es|
        dependencies inside the environment, or only the subset of
        dependencies not already satisfied by your workstation or cluster.
        For the exact syntax to create and configure an environment,
        please refer to the tool documentation.

    Cython
        Cython is used for connecting the C++ core to Python.

        Python environment tools may allow you to install a Python executable
        that is more recent than the system-wide Python executable.
        Be aware this might lead to compatibility issues if Cython
        accidentally picks up the system-wide :file:`Python.h` header file.
        In that scenario, you will have to manually adapt the C++ compiler
        include paths to find the correct :file:`Python.h` header file.


.. _Installing requirements on Ubuntu Linux:

Installing requirements on Ubuntu Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compile |es| on Ubuntu 22.04 LTS, install the following dependencies:

.. code-block:: bash

    sudo apt install build-essential cmake cython3 python3-dev openmpi-bin \
      libboost-all-dev fftw3-dev libfftw3-mpi-dev libhdf5-dev libhdf5-openmpi-dev \
      python3-pip python3-numpy python3-scipy python3-opengl libgsl-dev freeglut3

Optionally the ccmake utility can be installed for easier configuration:

.. code-block:: bash

    sudo apt install cmake-curses-gui

.. _Nvidia GPU acceleration:

Nvidia GPU acceleration
"""""""""""""""""""""""

If your computer has an Nvidia graphics card, you should also download and install the
CUDA SDK to make use of GPU computation:

.. code-block:: bash

    sudo apt install nvidia-cuda-toolkit

Later in the installation instructions, you will see CMake commands of the
form ``cmake ..`` with optional arguments, such as ``cmake .. -D WITH_CUDA=ON``
to activate CUDA. These commands may need to be adapted depending on which
operating system and CUDA version you are using.

On Ubuntu 22.04, the default GCC compiler is too recent for nvcc and will fail
to compile sources that rely on ``std::function``. You can either use GCC 10:

.. code-block:: bash

    CC=gcc-10 CXX=g++-10 cmake .. -D WITH_CUDA=ON

or alternatively install Clang 12 as a replacement for nvcc and GCC:

.. code-block:: bash

    CC=clang-12 CXX=clang++-12 cmake .. -D WITH_CUDA=ON -D WITH_CUDA_COMPILER=clang

On Ubuntu 20.04, the default GCC compiler is also too recent for nvcc and will
generate compiler errors. You can either install an older version of GCC and
select it with environment variables ``CC`` and ``CXX`` when building |es|,
or edit the system header files as shown in the following
`patch for Ubuntu 20.04 <https://github.com/espressomd/espresso/issues/3654#issuecomment-612165048>`__.

.. _Requirements for building the documentation:

Requirements for building the documentation
"""""""""""""""""""""""""""""""""""""""""""

To generate the Sphinx documentation, install the following packages:

.. code-block:: bash

    pip3 install --user -c requirements.txt \
        sphinx sphinxcontrib-bibtex sphinx-toggleprompt

To generate the Doxygen documentation, install the following packages:

.. code-block:: bash

    sudo apt install doxygen graphviz

.. _Setting up a Jupyter environment:

Setting up a Jupyter environment
""""""""""""""""""""""""""""""""

To run the samples and tutorials, start by installing the following packages:

.. code-block:: bash

    sudo apt install python3-matplotlib python3-pint python3-tqdm ffmpeg
    pip3 install --user 'MDAnalysis>=1.0.0,<2.0.0'

The tutorials are written in the
`Notebook Format <https://nbformat.readthedocs.io/en/latest/>`__
version <= 4.4 and can be executed by any of these tools:

* `Jupyter Notebook <https://jupyter-notebook.readthedocs.io/en/stable/notebook.html>`__
* `JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`__
* `IPython <https://ipython.org/>`__ (not recommended)
* `VS Code Jupyter <https://github.com/microsoft/vscode-jupyter>`__

To check whether one of them is installed, run these commands:

.. code-block:: bash

    jupyter notebook --version
    jupyter lab --version
    ipython --version
    code --version

If you don't have any of these tools installed and aren't sure which one
to use, we recommend installing the historic Jupyter Notebook, since the
|es| tutorials have been designed with the ``exercise2`` plugin in mind.

To use Jupyter Notebook, install the following packages:

.. code-block:: bash

    pip3 install --user 'nbformat==5.1.3' 'nbconvert==6.4.5' 'notebook==6.4.8' 'jupyter_contrib_nbextensions==0.5.1'
    jupyter contrib nbextension install --user
    jupyter nbextension enable rubberband/main
    jupyter nbextension enable exercise2/main

Alternatively, to use JupyterLab, install the following packages:

.. code-block:: bash

    pip3 install --user nbformat notebook jupyterlab

Alternatively, to use VS Code Jupyter, install the following extensions:

.. code-block:: bash

    code --install-extension ms-python.python
    code --install-extension ms-toolsai.jupyter
    code --install-extension ms-toolsai.jupyter-keymap
    code --install-extension ms-toolsai.jupyter-renderers

.. _Installing requirements on other Linux distributions:

Installing requirements on other Linux distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please refer to the following Dockerfiles to find the minimum set of packages
required to compile |es| on other Linux distributions:

* `Fedora <https://github.com/espressomd/docker/blob/4.2/docker/Dockerfile-fedora>`__
* `Debian <https://github.com/espressomd/docker/blob/4.2/docker/Dockerfile-debian>`__

.. _Installing requirements on Windows via WSL:

Installing requirements on Windows via WSL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run |es| on Windows, use the Linux subsystem. For that you need to

* follow `these instructions <https://learn.microsoft.com/en-us/windows/wsl/install>`__ to install Ubuntu
* start Ubuntu (or open an Ubuntu tab in `Windows Terminal <https://apps.microsoft.com/detail/9n0dx20hk701?hl=en-us&gl=US>`__)
* execute ``sudo apt update`` to prepare the installation of dependencies
* optional step: If you have a NVIDIA graphics card available and want to make
  use of |es|'s GPU acceleration, follow `these instructions <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`__
  to set up CUDA.
* follow the instructions for :ref:`Installing requirements on Ubuntu Linux`

.. _Installing requirements on macOS:

Installing requirements on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build |es| on macOS 10.15 or higher, you need to install its dependencies.
There are two possibilities for this, MacPorts and Homebrew. We strongly
recommend Homebrew, but if you already have MacPorts installed, you can use
that too, although we do not provide MacPorts installation instructions.

To check whether you already have one or the other installed, run the
following commands:

.. code-block:: bash

    test -e /opt/local/bin/port && echo "MacPorts is installed"
    test -e /usr/local/bin/brew && echo "Homebrew is installed"

If Homebrew is already installed, you should resolve any problems reported by
the command

.. code-block:: bash

    brew doctor

If you want to install Homebrew, follow the installation instructions at
https://docs.brew.sh/Installation, but bear in mind that MacPorts and Homebrew
may conflict with one another.

If Anaconda Python or the Python from www.python.org are installed, you
will likely not be able to run |es|. Therefore, please uninstall them
using the following commands:

.. code-block:: bash

    sudo rm -r ~/anaconda[23]
    sudo rm -r /Library/Python

Installing packages using Homebrew
""""""""""""""""""""""""""""""""""

Run the following commands:

.. code-block:: bash

    brew install cmake python cython boost boost-mpi fftw \
      doxygen gsl numpy scipy ipython jupyter freeglut
    brew install hdf5-mpi
    brew link --force cython
    pip install -c requirements.txt PyOpenGL matplotlib

.. _Quick installation:

Quick installation
------------------

If you have installed the requirements (see section :ref:`Requirements`) in
standard locations, compiling |es| is usually only a matter of creating a build
directory and calling ``cmake`` and ``make`` in it. See for example the command
lines below (optional steps which modify the build process are commented out):

.. code-block:: bash

    mkdir build
    cd build
    cmake ..
    #ccmake . // in order to add/remove features like ScaFaCoS or CUDA
    make -j

This will build |es| with a default feature set, namely
:file:`src/config/myconfig-default.hpp`. This file is a C++ header file,
which defines the features that should be compiled in.
You may want to adjust the feature set to your needs. This can be easily
done by copying the :file:`myconfig-sample.hpp` which has been created in
the :file:`build` directory to :file:`myconfig.hpp` and only uncomment
the features you want to use in your simulation.

The ``cmake`` command looks for libraries and tools needed by |es|.
So |es| can only be built if ``cmake`` reports no errors.

The command ``make`` will compile the source code. Depending on the
options passed to the program, ``make`` can also be used for a number of
other things:

*  It can install and uninstall the program to some other directories.
   However, normally it is not necessary to actually *install* to run
   it: ``make install``

*  It can invoke code checks: ``make check``

*  It can build this documentation: ``make sphinx``

When these steps have successfully completed, |es| can be started with the
command:

.. code-block:: bash

    ./pypresso script.py

where ``script.py`` is a Python script which has to be written by the user.
You can find some examples in the :file:`samples` folder of the source code
directory. If you want to run in parallel, you should have compiled with an
MPI library, and need to tell MPI to run in parallel.
The actual invocation is implementation-dependent, but in many cases, such as
*Open MPI* and *MPICH*, you can use

.. code-block:: bash

    mpirun -n 4 ./pypresso script.py

where ``4`` is the number of processors to be used.


.. _Features:

Features
--------

This chapter describes the features that can be activated in |es|. Even if
possible, it is not recommended to activate all features, because this
will negatively affect |es|'s performance.

Features can be activated in the configuration header :file:`myconfig.hpp`
(see section :ref:`myconfig.hpp\: Activating and deactivating features`).
To activate ``FEATURE``, add the following line to the header file:

.. code-block:: c++

    #define FEATURE


.. _General features:

General features
~~~~~~~~~~~~~~~~

-  ``ELECTROSTATICS`` This enables the use of the various electrostatics algorithms, such as P3M.

   .. seealso:: :ref:`Electrostatics`

-  ``MMM1D_GPU``: This enables MMM1D on GPU. It is faster than the CPU version
   by several orders of magnitude, but has float precision instead of double
   precision.

-  ``MMM1D_MACHINE_PREC``: This enables high-precision Bessel functions
   for MMM1D on CPU. Comes with a 60% slow-down penalty. The low-precision
   functions are enabled by default and are precise enough for most applications.

-  ``DIPOLES`` This activates the dipole-moment property of particles and switches
   on various magnetostatics algorithms

   .. seealso:: :ref:`Magnetostatics`

-  ``SCAFACOS_DIPOLES`` This activates magnetostatics methods of ScaFaCoS.

-  ``DIPOLAR_DIRECT_SUM`` This activates the GPU implementation of the dipolar direct sum.

-  ``ROTATION`` Switch on rotational degrees of freedom for the particles, as well as
   the corresponding quaternion integrator.

   .. seealso:: :ref:`Setting up particles`

   .. note::
      When this feature is activated, every particle has three
      additional degrees of freedom, which for example means that the
      kinetic energy changes at constant temperature is twice as large.

-  ``THERMOSTAT_PER_PARTICLE`` Allows setting a per-particle friction
   coefficient for the Langevin and Brownian thermostats.

-  ``ROTATIONAL_INERTIA`` Allows particles to have individual rotational inertia matrix eigenvalues.
   When not built in, all eigenvalues are unity in simulation units.

-  ``EXTERNAL_FORCES`` Allows to define an arbitrary constant force for each particle
   individually. Also allows to fix individual coordinates of particles,
   keep them at a fixed position or within a plane.

-  ``MASS`` Allows particles to have individual masses.
   When not built in, all masses are unity in simulation units.

   .. seealso:: :attr:`espressomd.particle_data.ParticleHandle.mass`

-  ``EXCLUSIONS`` Allows to exclude specific short ranged interactions within
   molecules.

   .. seealso:: :meth:`espressomd.particle_data.ParticleHandle.add_exclusion`

-  ``BOND_CONSTRAINT`` Turns on the RATTLE integrator which allows for fixed lengths bonds
   between particles.

-  ``VIRTUAL_SITES_RELATIVE`` Virtual sites are particles, the position and velocity of which is
   not obtained by integrating equations of motion. Rather, they are
   placed using the position (and orientation) of other particles. The
   feature allows for rigid arrangements of particles.

   .. seealso:: :ref:`Virtual sites`

-  ``COLLISION_DETECTION`` Allows particles to be bound on collision.

In addition, there are switches that enable additional features in the
integrator or thermostat:

-  ``NPT`` Enables an on-the-fly NpT integration scheme.

   .. seealso:: :ref:`Isotropic NpT thermostat`

-  ``ENGINE``

-  ``PARTICLE_ANISOTROPY``


.. _Fluid dynamics and fluid structure interaction:

Fluid dynamics and fluid structure interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ``DPD`` Enables the dissipative particle dynamics thermostat and interaction.

   .. seealso:: :ref:`DPD interaction`

-  ``LB_BOUNDARIES``

-  ``LB_BOUNDARIES_GPU``

-  ``LB_ELECTROHYDRODYNAMICS`` Enables the implicit calculation of electro-hydrodynamics for charged
   particles and salt ions in an electric field.

-  ``ELECTROKINETICS``

-  ``EK_BOUNDARIES``

-  ``EK_DEBUG``


.. _Interaction features:

Interaction features
~~~~~~~~~~~~~~~~~~~~

The following switches turn on various short ranged interactions (see
section :ref:`Isotropic non-bonded interactions`):

-  ``TABULATED`` Enable support for user-defined non-bonded interaction potentials.

-  ``LENNARD_JONES`` Enable the Lennard-Jones potential.

-  ``LENNARD_JONES_GENERIC`` Enable the generic Lennard-Jones potential with configurable
   exponents and individual prefactors for the two terms.

-  ``LJCOS`` Enable the Lennard-Jones potential with a cosine-tail.

-  ``LJCOS2`` Same as ``LJCOS``, but using a slightly different way of smoothing the
   connection to 0.

-  ``WCA`` Enable the Weeks--Chandler--Andersen potential.

-  ``GAY_BERNE`` Enable the Gay--Berne potential.

-  ``HERTZIAN`` Enable the Hertzian potential.

-  ``MORSE`` Enable the Morse potential.

-  ``BUCKINGHAM`` Enable the Buckingham potential.

-  ``SOFT_SPHERE`` Enable the soft sphere potential.

-  ``SMOOTH_STEP`` Enable the smooth step potential, a step potential with
   two length scales.

-  ``BMHTF_NACL`` Enable the Born--Meyer--Huggins--Tosi--Fumi potential,
   which can be used to model salt melts.

-  ``GAUSSIAN`` Enable the Gaussian potential.

-  ``HAT`` Enable the Hat potential.

Some of the short-range interactions have additional features:

-  ``LJGEN_SOFTCORE`` This modifies the generic Lennard-Jones potential
   (``LENNARD_JONES_GENERIC``) with tunable parameters.

-  ``THOLE`` See :ref:`Thole correction`


.. _Debug messages:

Debug messages
~~~~~~~~~~~~~~

Finally, there is a flag for debugging:

-  ``ADDITIONAL_CHECKS`` Enables numerous additional checks which can detect
   inconsistencies especially in the cell systems. These checks are however
   too slow to be enabled in production runs.

   .. note::
      Because of a bug in OpenMPI versions 2.0-2.1, 3.0.0-3.0.2 and 3.1.0-3.1.2
      that causes a segmentation fault when running the |es| OpenGL visualizer
      with feature ``ADDITIONAL_CHECKS`` enabled together with either
      ``ELECTROSTATICS`` or ``DIPOLES``, the subset of additional checks for
      those two features are disabled if an unpatched version of OpenMPI is
      detected during compilation.


.. _External features:

External features
~~~~~~~~~~~~~~~~~

External features cannot be added to the :file:`myconfig.hpp` file by the user.
They are added by CMake if the corresponding dependency was found on the
system. Some of these external features are optional and must be activated
using a CMake flag (see :ref:`Options and Variables`).

- ``CUDA`` Enables GPU-specific features.

- ``FFTW`` Enables features relying on the fast Fourier transforms, e.g. P3M.

- ``H5MD`` Write data to H5MD-formatted hdf5 files (see :ref:`Writing H5MD-files`)

- ``SCAFACOS`` Enables features relying on the ScaFaCoS library (see
  :ref:`ScaFaCoS electrostatics`, :ref:`ScaFaCoS magnetostatics`).

- ``GSL`` Enables features relying on the GNU Scientific Library, e.g.
  :meth:`espressomd.cluster_analysis.Cluster.fractal_dimension`.

- ``STOKESIAN_DYNAMICS`` Enables the Stokesian Dynamics feature
  (see :ref:`Stokesian Dynamics`). Requires BLAS and LAPACK.



.. _Configuring:

Configuring
-----------

.. _myconfig.hpp\: Activating and deactivating features:

:file:`myconfig.hpp`: Activating and deactivating features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|es| has a large number of features that can be compiled into the binary.
However, it is not recommended to actually compile in all possible
features, as this will slow down |es| significantly. Instead, compile in only
the features that are actually required. A strong gain in speed can be
achieved by disabling all non-bonded interactions except for a single
one, e.g. ``LENNARD_JONES``. For developers, it is also possible to turn on or off a
number of debugging messages. The features and debug messages can be
controlled via a configuration header file that contains C-preprocessor
declarations. Subsection :ref:`Features` describes all available features. If a
file named :file:`myconfig.hpp` is present in the build directory when ``cmake``
is run, all features defined in it will be compiled in. If no such file exists,
the configuration file :file:`src/config/myconfig-default.hpp` will be used
instead, which turns on the default features.

When you distinguish between the build and the source directory, the
configuration header can be put in either of these. Note, however, that
when a configuration header is found in both directories, the one in the
build directory will be used.

By default, the configuration header is called :file:`myconfig.hpp`.
The configuration header can be used to compile different binary
versions of with a different set of features from the same source
directory. Suppose that you have a source directory :file:`$srcdir` and two
build directories :file:`$builddir1` and :file:`$builddir2` that contain
different configuration headers:

* :file:`$builddir1/myconfig.hpp`:

  .. code-block:: c++

      #define ELECTROSTATICS
      #define LENNARD_JONES

* :file:`$builddir2/myconfig.hpp`:

  .. code-block:: c++

      #define LJCOS

Then you can simply compile two different versions of |es| via:

.. code-block:: bash

    cd $builddir1
    cmake ..
    make

    cd $builddir2
    cmake ..
    make

To see what features were activated in :file:`myconfig.hpp`, run:

.. code-block:: bash

    ./pypresso

and then in the Python interpreter:

.. code-block:: python

    import espressomd
    print(espressomd.features())


.. _cmake:

``cmake``
~~~~~~~~~

In order to build the first step is to create a build directory in which
cmake can be executed. In cmake, the *source directory* (that contains
all the source files) is completely separated from the *build directory*
(where the files created by the build process are put). ``cmake`` is
designed to *not* be executed in the source directory. ``cmake`` will
determine how to use and where to find the compiler, as well as the
different libraries and tools required by the compilation process. By
having multiple build directories you can build several variants of |es|,
each variant having different activated features, and for as many
platforms as you want.

Once you've run ``ccmake``, you can list the configured variables with
``cmake -LAH -N . | less`` (uses a pager) or with ``ccmake ..`` and pressing
key ``t`` to toggle the advanced mode on (uses the ``curses`` interface).

**Example:**

When the source directory is :file:`srcdir` (the files where unpacked to this
directory), then the user can create a build directory :file:`build` below that
path by calling ``mkdir srcdir/build``. In the build directory ``cmake`` is to be
executed, followed by a call to ``make``. None of the files in the source directory
are ever modified by the build process.

.. code-block:: bash

    cd build
    cmake ..
    make -j

Afterwards |es| can be run by calling ``./pypresso`` from the command line.


.. _ccmake:

``ccmake``
~~~~~~~~~~

Optionally and for easier use, the curses interface to cmake can be used
to configure |es| interactively.

**Example:**

Alternatively to the previous example, instead of cmake, the ccmake executable
is called in the build directory to configure |es|, followed by a call to make:

.. code-block:: bash

    cd build
    ccmake ..
    make

Fig. :ref:`ccmake-figure` shows the interactive ccmake UI.

.. _ccmake-figure:

.. figure:: figures/ccmake-example.png
   :alt: ccmake interface
   :width: 70.0%
   :align: center

   ccmake interface


.. _Options and Variables:

Options and Variables
~~~~~~~~~~~~~~~~~~~~~

The behavior of |es| can be controlled by means of options and variables
in the :file:`CMakeLists.txt` file. Most options are boolean values
(``ON`` or ``OFF``). A few options are strings or semicolon-delimited lists.

The following options control features from external libraries:

* ``WITH_CUDA``: Build with GPU support.
* ``WITH_HDF5``: Build with HDF5 support.
* ``WITH_SCAFACOS``: Build with ScaFaCoS support.
* ``WITH_GSL``: Build with GSL support.
* ``WITH_STOKESIAN_DYNAMICS`` Build with Stokesian Dynamics support.
* ``WITH_PYTHON`` Build with Stokesian Dynamics support.

The following options control code instrumentation:

* ``WITH_VALGRIND_INSTRUMENTATION``: Build with valgrind instrumentation markers
* ``WITH_PROFILER``: Build with Caliper profiler annotations
* ``WITH_MSAN``: Compile C++ code with memory sanitizer
* ``WITH_ASAN``: Compile C++ code with address sanitizer
* ``WITH_UBSAN``: Compile C++ code with undefined behavior sanitizer
* ``WITH_COVERAGE``: Generate C++ code coverage reports when running |es|
* ``WITH_COVERAGE_PYTHON``: Generate Python code coverage reports when running |es|

The following options control how the project is built and tested:

* ``WITH_CLANG_TIDY``: Run Clang-Tidy during compilation.
* ``WITH_CPPCHECK``: Run Cppcheck during compilation.
* ``WITH_CCACHE``: Enable compiler cache for faster rebuilds.
* ``WITH_TESTS``: Enable C++ and Python tests.
* ``WITH_BENCHMARKS``: Enable benchmarks.
* ``WITH_CUDA_COMPILER`` (string): Select the CUDA compiler.
* ``CTEST_ARGS`` (string): Arguments passed to the ``ctest`` command.
* ``TEST_TIMEOUT``: Test timeout.
* ``ESPRESSO_ADD_OMPI_SINGLETON_WARNING``: Add a runtime warning in the
  pypresso and ipypresso scripts that is triggered in singleton mode
  with Open MPI version 4.x on unsupported NUMA environments
  (see :term:`MPI installation requirements <MPI>` for details).
* ``MYCONFIG_NAME`` (string): Filename of the user-provided config file
* ``MPIEXEC_PREFLAGS``, ``MPIEXEC_POSTFLAGS`` (strings): Flags passed to the
  ``mpiexec`` command in MPI-parallel tests and benchmarks.
* ``CMAKE_CXX_FLAGS`` (string): Flags passed to the compilers.
* ``CMAKE_BUILD_TYPE`` (string): Build type. Default is ``Release``.
* ``CUDA_TOOLKIT_ROOT_DIR`` (string): Path to the CUDA toolkit directory.

Most of these options are opt-in, meaning their default value is set to
``OFF`` in the :file:`CMakeLists.txt` file. These options can be modified
by calling ``cmake`` with the command line argument ``-D``:

.. code-block:: bash

    cmake -D WITH_HDF5=OFF ..

When an option is enabled, additional options may become available.
For example with ``-D WITH_CUDA=ON``, one can choose the CUDA compiler with
``-D WITH_CUDA_COMPILER=<compiler_id>``, where ``<compiler_id>`` can be
``nvcc`` (default) or ``clang``.

Environment variables can be passed to CMake. For example, to select Clang, use
``CC=clang CXX=clang++ cmake .. -DWITH_CUDA=ON -DWITH_CUDA_COMPILER=clang``.
If you have multiple versions of the CUDA library installed, you can select the
correct one with ``CUDA_BIN_PATH=/usr/local/cuda-10.0 cmake .. -DWITH_CUDA=ON``
(with Clang as the CUDA compiler, you also need to override its default CUDA
path with ``-DCMAKE_CXX_FLAGS=--cuda-path=/usr/local/cuda-10.0``).

.. _Build types and compiler flags:

Build types and compiler flags
""""""""""""""""""""""""""""""

The build type is controlled by ``-D CMAKE_BUILD_TYPE=<type>`` where
``<type>`` can take one of the following values:

* ``Release``: for production use: disables assertions and debug information,
  enables ``-O3`` optimization (this is the default)
* ``RelWithAssert``: for debugging purposes: enables assertions and
  ``-O3`` optimization (use this to track the source of a fatal error)
* ``Debug``: for debugging in GDB
* ``Coverage``: for code coverage

Cluster users and HPC developers may be interested in manually editing the
``Espresso_cpp_flags`` target in the top-level ``CMakeLists.txt`` file for
finer control over compiler flags. The variable declaration is followed
by a series of conditionals to enable or disable compiler-specific flags.
Compiler flags passed to CMake via the ``-DCMAKE_CXX_FLAGS`` option
(such as ``cmake . -DCMAKE_CXX_FLAGS="-ffast-math -fno-finite-math-only"``)
will appear in the compiler command before the flags in ``Espresso_cpp_flags``,
and will therefore have lower precedence.

Be aware that fast-math mode can break |es|. It is incompatible with the
``ADDITIONAL_CHECKS`` feature due to the loss of precision in the LB code
on CPU. The Clang 10 compiler breaks field couplings with ``-ffast-math``.
The Intel compiler enables the ``-fp-model fast=1`` flag by default;
it can be disabled by adding the ``-fp-model=strict`` flag.

|es| currently doesn't fully support link-time optimization (LTO).


.. _Configuring without a network connection:

Configuring without a network connection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several :ref:`external features <External features>` in |es| rely on
external libraries that are downloaded automatically by CMake. When a
network connection cannot be established due to firewall restrictions,
the CMake logic needs editing:

* ``WITH_HDF5``: when cloning |es|, the :file:`libs/h5xx` folder will be
  a git submodule containing a :file:`.git` subfolder. To prevent CMake from
  updating this submodule with git, delete the corresponding command with:

  .. code-block:: bash

    sed -i '/execute_process(COMMAND ${GIT_EXECUTABLE} submodule update -- libs\/h5xx/,+1 d' CMakeLists.txt

  When installing a release version of |es|, no network communication
  is needed for HDF5.

* ``WITH_STOKESIAN_DYNAMICS``: this library is installed using `FetchContent
  <https://cmake.org/cmake/help/latest/module/FetchContent.html>`__.
  The repository URL can be found in the ``GIT_REPOSITORY`` field of the
  corresponding ``FetchContent_Declare()`` command. The ``GIT_TAG`` field
  provides the commit. Clone this repository locally next to the |es|
  folder and edit the |es| build system such that ``GIT_REPOSITORY`` points
  to the absolute path of the Stokesian Dynamics clone, for example with:

  .. code-block:: bash

    sed -ri 's|GIT_REPOSITORY +.+stokesian-dynamics.git|GIT_REPOSITORY /work/username/stokesian_dynamics|' CMakeLists.txt


Compiling, testing and installing
---------------------------------

The command ``make`` is mainly used to compile the source code, but it
can do a number of other things. The generic syntax of the ``make``
command is:

.. code-block:: bash

    make [options] [target] [variable=value]

When no target is given, the target ``all`` is used. The following
targets are available:

``all``
    Compiles the complete source code. The variable can be used to
    specify the name of the configuration header to be used.

``check``
    Runs the testsuite. By default, all available tests will be run on
    1, 2, 3, 4, 6, or 8 processors.

``test``
    Do not use this target, it is a broken feature
    (see `issue #4370 <https://github.com/espressomd/espresso/issues/4370>`__).
    Use ``make check`` instead.

``clean``
    Deletes all files that were created during the compilation.

``install``
    Install |es| in the path specified by the CMake variable
    ``CMAKE_INSTALL_PREFIX``. The path can be changed by calling CMake
    with ``cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/espresso``. Do not use
    ``make DESTDIR=/path/to/espresso install`` to install to a specific path,
    this will cause issues with the runtime path (RPATH) and will conflict
    with the CMake variable ``CMAKE_INSTALL_PREFIX`` if it has been set.

``doxygen``
    Creates the Doxygen code documentation in the :file:`doc/doxygen`
    subdirectory.

``sphinx``
    Creates the ``sphinx`` code documentation in the :file:`doc/sphinx`
    subdirectory.

``tutorials``
    Creates the tutorials in the :file:`doc/tutorials` subdirectory.

``doc``
    Creates all documentation in the :file:`doc` subdirectory (only when
    using the development sources).

A number of options are available when calling ``make``. The most
interesting option is probably ``-j num_jobs``, which can be used for
parallel compilation. ``num_jobs`` specifies the maximal number of
concurrent jobs that will be run. Setting ``num_jobs`` to the number
of available processors speeds up the compilation process significantly.

.. _Troubleshooting:

Troubleshooting
---------------

If you encounter issues when building |es| or running it for the first time,
please have a look at the `Installation FAQ <https://github.com/espressomd/espresso/wiki/Installation-FAQ>`__
on the wiki. If you still didn't find an answer, see :ref:`Community support`.

Many algorithms require parameters that must be provided within valid ranges.
Range checks are implemented to catch invalid input values and generate
meaningful error messages, however these checks cannot always catch errors
arising from an invalid combination of two or more features. If you encounter
issues with a script, you can activate extra runtime checks by enabling C++
assertions. This is achieved by updating the CMake project and rebuilding
|es| with:

.. code-block:: bash

    cmake . -DCMAKE_BUILD_TYPE=RelWithAssert
    make -j

The resulting build will run slightly slower, but will produce an error
message for common issues, such as divisions by zero, array access out
of bounds, or square roots of negative numbers.

If this still doesn't help, you can activate debug symbols to help with
instrumentation:

.. code-block:: bash

    cmake . -DCMAKE_BUILD_TYPE=Debug
    make -j

The resulting build will be quite slow but will allow many debugging tools
to be used. For details, please refer to chapter :ref:`Debugging es`.

If you are dealing with a segmentation fault or undefined behavior, and GDB
doesn't help or is too cumbersome to use (e.g. in MPI-parallel simulations),
you can as a last resort activate sanitizers:

.. code-block:: bash

    cmake . -DWITH_ASAN=ON -DWITH_UBSAN=ON -DCMAKE_BUILD_TYPE=Release
    make -j

The resulting build will be around 5 times slower that a debug build,
but it will generate valuable reports when detecting fatal exceptions.
For more details, please consult the online documentation of
`UBSAN <https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html>`__ and
`ASAN <https://github.com/google/sanitizers/wiki/AddressSanitizer>`__.

____

.. [1]
   https://espressomd.org

.. [2]
   https://git-scm.com/

.. [3]
   https://github.com/espressomd/espresso

.. [4]
   https://cmake.org/

.. [5]
   https://www.fftw.org/

.. [6]
   https://docs.nvidia.com/cuda/

.. [7]
   https://docs.python.org/3/library/venv.html

.. [8]
   https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

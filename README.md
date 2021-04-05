<p align=center>
    <a target="_blank" href="https://www.python.org/downloads/" title="Python version"><img src="https://img.shields.io/badge/python-%3E=_3.6-green.svg"></a>
    <a target="_blank" href="https://pypi.org/project/cutde/" title="PyPI version"><img src="https://img.shields.io/pypi/v/cutde?logo=pypi"></a>
    <!-- <a target="_blank" href="https://pypi.org/project/cutde/" title="PyPI"><img src="https://img.shields.io/pypi/dm/cutde"></a> -->
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a target="_blank" href="https://github.com/tbenthompson/cutde/actions" title="Test Status"><img src="https://github.com/tbenthompson/cutde/actions/workflows/test.yml/badge.svg"></a>
</a>

# Python + CUDA TDEs from Nikkhoo and Walter 2015

CUDA and OpenCL-enabled fullspace triangle dislocation elements. Benchmarked at 130 million TDEs per second. Based on the [original MATLAB code from Nikhoo and Walter 2015.](https://volcanodeformation.com/software)

See below for usage and installation instructions.

<!--ts-->
   * [Python + CUDA TDEs from Nikkhoo and Walter 2015](#python--cuda-tdes-from-nikkhoo-and-walter-2015)
   * [Usage documentation](#usage-documentation)
      * [I want stress.](#i-want-stress)
      * [All pairs](#all-pairs)
   * [Installation](#installation)
      * [PyCUDA](#pycuda)
      * [Mac OS X](#mac-os-x)
      * [Ubuntu + PyOpenCL/PoCL](#ubuntu--pyopenclpocl)
      * [Ubuntu + PyOpenCL with system drivers**](#ubuntu--pyopencl-with-system-drivers)
      * [Windows](#windows)
      * [Something else](#something-else)
      * [Why can't I use Apple CPU OpenCL?](#why-cant-i-use-apple-cpu-opencl)
   * [Development](#development)

<!-- Added by: tbent, at: Mon 05 Apr 2021 05:15:11 PM EDT -->

<!--te-->

```python

import matplotlib.pyplot as plt
import numpy as np

import cutde

xs = np.linspace(-2, 2, 200)
ys = np.linspace(-2, 2, 200)
obsx, obsy = np.meshgrid(xs, ys)
pts = np.array([obsx, obsy, 0 * obsy]).reshape((3, -1)).T.copy()

fault_pts = np.array([[-1, 0, 0], [1, 0, 0], [1, 0, -1], [-1, 0, -1]])
fault_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

#
slip = np.array([[1, 0, 0], [1, 0, 0]])

disp_mat = cutde.disp_all_pairs(
    obs_pts=pts, tris=fault_pts[fault_tris], slips=slip, nu=0.25
)
disp = np.sum(disp_mat, axis=1).reshape((*obsx.shape, 3))

plt.figure(figsize=(5, 5), dpi=300)
cntf = plt.contourf(obsx, obsy, disp[:, :, 0], levels=21)
plt.contour(
    obsx, obsy, disp[:, :, 0], colors="k", linestyles="-", linewidths=0.5, levels=21
)
plt.colorbar(cntf)
plt.title("$u_x$")
plt.tight_layout()
plt.savefig("docs/example.png", bbox_inches="tight")

```

![docs/example.png](docs/example.png)

# Usage documentation

Usage is really simple:

```
import cutde

disp = cutde.disp(pts, tris, slips, 0.25)
strain = cutde.strain(pts, tris, slips, nu)
```

* `pts` is a `np.array` with shape `(N, 3)`
* tris is a `np.array` with shape `(N, 3, 3)` where the second dimension corresponds to each vertex and the third dimension corresponds to the cooordinates of those vertices.
* slips is a `np.array` with shape `(N, 3)` where `slips[:,0]` is the strike slip component, while component 1 is the dip slip and component 2 is the tensile/opening component.
* the last parameter, nu, is the Poisson ratio. 

IMPORTANT: N should be the same for all these arrays. There is exactly one triangle and slip value used for each observation point. 

* The output `disp` is a `(N, 3)` array with displacement components in the x, y, z directions.
* The output `strain` is a `(N, 6)` array representing a symmetric tensor. `strain[:,0]` is the xx component of strain, 1 is yy, 2 is zz, 3 is xy, 4 is xz, and 5 is  yz.

## I want stress.

Use:

```
stress = cutde.strain_to_stress(strain, sm, nu)
```

to convert from stress to strain assuming isotropic linear elasticity. `sm` is the shear modulus and `nu` is the Poisson ratio.

## All pairs

If, instead, you want to create a matrix representing the interaction between every observation point and every source triangle, there is a different interface:

```
import cutde

disp = cutde.disp_all_pairs(pts, tris, slips, 0.25)
strain = cutde.strain_all_pairs(pts, tris, slips, nu)
```

* `pts` is a `np.array` with shape `(N_OBS_PTS, 3)`
* tris is a `np.array` with shape `(N_SRC_TRIS, 3, 3)` where the second dimension corresponds to each vertex and the third dimension corresponds to the cooordinates of those vertices.
* slips is a `np.array` with shape `(N_SRC_TRIS, 3)` where `slips[:,0]` is the strike slip component, while component 1 is the dip slip and component 2 is the tensile/opening component.
* the last parameter, nu, is the Poisson ratio. 
* The output `disp` is a `(N_OBS_PTS, N_SRC_TRIS, 3)` array.
* The output `strain` is a `(N_OBS_PTS, N_SRC_TRIS, 6)` array.

Note that to use the `strain_to_stress` function, you'll need to reshape the output strain to be `(N_OBS_PTS * N_SRC_TRIS, 6)`.

# Installation

To install `cutde` itself run:
```
pip install cutde
```

Then, install either PyCUDA or PyOpenCL following the directions below.

## PyCUDA
If you have an NVIDIA GPU, install PyCUDA with:
```
conda config --prepend channels conda-forge
conda install -c conda-forge pycuda
```

## Mac OS X
Install PyOpenCL and the PoCL OpenCL driver with:
```
conda config --prepend channels conda-forge
conda install pocl pyopencl
```

## Ubuntu + PyOpenCL/PoCL

Just like on a Mac:
```
conda config --prepend channels conda-forge
conda install pocl pyopencl
```


## Ubuntu + PyOpenCL with system drivers** 
```
conda install pyopencl ocl-icd ocl-icd-system
```
You will need to install the system OpenCL drivers yourself depending on the hardware you have. See the "Something else" section below.

## Windows

I'm not aware of anyone testing cutde on Windows yet. It should not be difficult to install. I would expect that you install pyopencl via conda and then install the OpenCL libraries and drivers that are provided by your hardware vendor. See the "Something else" section below.

## Something else
I'd suggest starting by trying the instructions for the system most similar to yours above. If that doesn't work, never fear! OpenCL should be installable on almost all recent hardware and typical operating systems. [These directions can be helpful.](https://documen.tician.de/pyopencl/misc.html#installing-from-conda-forge). I am happy to try to help if you have OpenCL installation issues, but I can't promise to be useful.

## Why can't I use Apple CPU OpenCL?

You might have gotten the message: `cutde does not support the Apple CPU OpenCL implementation and no other platform or device was found. Please consult the cutde README.`

The Apple OpenCL implementation for Intel CPUs has very poor support for the OpenCL standard and causes lots of difficult-to-resolve errors. Instead, please use [the PoCL implementation](http://portablecl.org/). You can install it with `conda install -c conda-forge pocl`.

# Development

For developing `cutde`, clone the repo and set up your conda environment based on the `environment.yml` with:

```
git clone https://github.com/tbenthompson/cutde.git
cd cutde
conda env create
conda activate cutde
pre-commit install
pip install --no-use-pep517 --disable-pip-version-check -e .
```

Next, install either `pycuda` or `pyopencl` as instructed in the Installation section above.

Then, you should re-generate the baseline test data derived from [the MATLAB code from Mehdi Nikhoo](https://volcanodeformation.com/software). To do this, first install `octave`. On Ubuntu, this is just:

```
sudo apt-get install octave
```

And run 

```
./tests/setup_test_env
```

which will run the `tests/matlab/gen_test_data.m` script.

Finally, to check that `cutde` is working properly, run `pytest`!

The library is extremely simple:
* `cutde.fullspace` - the main entrypoint.
* `fullspace.cu` - a direct translation of the original MATLAB into CUDA/OpenCL. This probably should not be modified.
* `cutde.gpu` - a layer that abstracts between CUDA and OpenCL
* `cutde.cuda` - the PyCUDA interface.
* `cutde.opencl` - the PyOpenCL interface.

The `tests/tde_profile.py` script is useful for assessing performance.

# cutde

<p align=center>
    <a target="_blank" href="https://www.python.org/downloads/" title="Python version"><img src="https://img.shields.io/badge/python-%3E=_3.6-green.svg"></a>
    <a target="_blank" href="https://pypi.org/project/cutde/" title="PyPI version"><img src="https://img.shields.io/pypi/v/cutde?logo=pypi"></a>
    <!-- <a target="_blank" href="https://pypi.org/project/cutde/" title="PyPI"><img src="https://img.shields.io/pypi/dm/cutde"></a> -->
    <a target="_blank" href="LICENSE" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</a>

# Python + CUDA TDEs from Nikkhoo and Walter 2015

CUDA and OpenCL-enabled fullspace triangle dislocation elements. Benchmarked at 130 million TDEs per second.

See below for usage and installation instructions.

### Basics

Howdy! Usage is really simple:

```
import cutde.fullspace

disp = cutde.fullspace.clu_disp(pts, tris, slips, 0.25)
strain = cutde.fullspace.clu_strain(pts, tris, slips, nu)
```

* `pts` is a `np.array` with shape `(N, 3)`
* tris is a `np.array` with shape `(N, 3, 3)` where the second dimension corresponds to each vertex and the third dimension corresponds to the cooordinates of those vertices.
* slips is a `np.array` with shape `(N, 3)` where `slips[:,0]` is the strike slip component, while component 1 is the dip slip and component 2 is the tensile/opening component.
* the last parameter, nu, is the Poisson ratio. 

IMPORTANT: N should be the same for all these arrays. There is exactly one triangle and slip value used for each observation point. 

* The output `disp` is a `(N, 3)` array with displacement components in the x, y, z directions.
* The output `strain` is a `(N, 6)` array representing a symmetric tensor. `strain[:,0]` is the xx component of strain, 1 is yy, 2 is zz, 3 is xy, 4 is xz, and 5 is  yz.

### I want stress.

Use:

```
stress = cutde.fullspace.strain_to_stress(strain, sm, nu)
```

to convert from stress to strain assuming isotropic linear elasticity. `sm` is the shear modulus and `nu` is the Poisson ratio.

### All pairs

If, instead, you want to create a matrix representing the interaction between every observation point and every source triangle, there is a different interface:

```
import cutde.fullspace

disp = cutde.fullspace.clu_disp_all_pairs(pts, tris, slips, 0.25)
strain = cutde.fullspace.clu_strain_all_pairs(pts, tris, slips, nu)
```

* `pts` is a `np.array` with shape `(N_OBS_PTS, 3)`
* tris is a `np.array` with shape `(N_SRC_TRIS, 3, 3)` where the second dimension corresponds to each vertex and the third dimension corresponds to the cooordinates of those vertices.
* slips is a `np.array` with shape `(N_SRC_TRIS, 3)` where `slips[:,0]` is the strike slip component, while component 1 is the dip slip and component 2 is the tensile/opening component.
* the last parameter, nu, is the Poisson ratio. 
* The output `disp` is a `(N_OBS_PTS, N_SRC_TRIS, 3)` array.
* The output `strain` is a `(N_OBS_PTS, N_SRC_TRIS, 6)` array.

Note that to use the `strain_to_stress` function, you'll need to reshape the output strain to be `(N_OBS_PTS * N_SRC_TRIS, 6)`.

### Installation

Just run 
```
pip install cutde
```

Then, if you have an NVIDIA GPU, install PyCUDA with:

```
conda install -c conda-forge pycuda
```

If not, you'll need to install PyOpenCL. Installing OpenCL is sometimes a breeze and sometimes a huge pain, but it should be installable on most recent hardware and typical operating systems. [These directions can be helpful.](https://documen.tician.de/pyopencl/misc.html#installing-from-conda-forge). I am happy to try to help if you have OpenCL installation issues, but I can't promise to be useful. For me, on an Ubuntu + Intel machine, I just ran:
```
conda install -c conda-forge pyopencl ocl-icd-system
```
and everything worked wonderfully.

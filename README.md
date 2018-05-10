# cutde
Python + CUDA TDEs from Nikkhoo and Walters 2015

Howdy! Usage is really simple:

```
import cutde.fullspace

disp = cutde.fullspace.clu_disp(pts, tris, slips, 0.25)
strain = cutde.fullspace.clu_strain(pts, tris, slips, nu)
```

where `pts` is a `np.array` with shape `(N, 3)`, tris is a `np.array` with shape `(N, 3, 3)`, 
slips is a `np.array` with shape `(N, 3)` and the last parameter is the Poisson ratio.

`slip[:,0]` is the strike slip component, while component 1 is the dip slip and component 2 is the tensile/opening component.

There is also a function

```
stress = cutde.fullspace.strain_to_stress(strain, sm, nu)
```

that converts from stress to strain assuming isotropic linear elasticity. `sm` is the shear modulus and `nu` is the Poisson ratio.

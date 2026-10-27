# CFD cases

2D lid-driven cavity flow solved with OpenFOAM `icoFoam` (incompressible, laminar). The lid velocity (U = 1 m/s) and the cavity size (L = 1 m) are fixed. The Reynolds number is set through the kinematic viscosity ν in `constant/transportProperties`.

## Contents

| Item | Purpose |
|---|---|
| `cavity_Re100`, `cavity_Re400`, `cavity_Re1000`, `cavity_Re2000` | Training data (80×80 mesh) |
| `cavity_Re700` | Test data (80×80 mesh), never used in training |
| `cavity_mesh20`, `cavity_mesh40`, `cavity_mesh80`, `cavity_mesh160` | Mesh independence study at Re = 1000 |
| `cavity_data.csv` | Cell-centre results of the Re cases (x, y, Re, Ux, Uy, p), used by the ML models |

## Running a case

The mesh (`constant/polyMesh`) and the solution time folders are not stored in the repository. To regenerate them, run inside a case folder:

```bash
cd cavity_Re1000
blockMesh
icoFoam
```

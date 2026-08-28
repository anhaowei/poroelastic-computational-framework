# Reproducing the LV poroelastic simulations

## Repository layout

Place the files in the following structure:

```text
project/
├── Singularity_fenics2019.img
├── LVMesh/
│   ├── ellipsoidal_baselinegeo.hdf5
│   └── ellipsoidal_baselinegeo_refine.hdf5
└── lv_poro_min_repo/
    ├── lvporo_min/
    ├── run_lv_case.py
    ├── postprocess.py
    └── outputs/
```

Run all commands from `lv_poro_min_repo`.

## Baseline simulation

The driver contains the equilibrated compartment volumes used to initialize
the production simulations. It runs ten complete 660-ms cardiac cycles by
default.

```bash
mkdir -p outputs

env \
  PORO_CASE_TYPE=baseline \
  PORO_CASE_ID=baseline_Rsin100_10cycles \
  PORO_R_SIN=100 \
  singularity exec ../Singularity_fenics2019.img \
  mpirun.mpich -launcher fork -np 6 python3 -u run_lv_case.py
```

## Coronary sinus occlusion simulations

Run each resistance in a separate output directory:

```bash
env PORO_CASE_TYPE=cso PORO_CASE_ID=cso_Rsin1e4_10cycles PORO_R_SIN=1e4 \
  singularity exec ../Singularity_fenics2019.img \
  mpirun.mpich -launcher fork -np 6 python3 -u run_lv_case.py

env PORO_CASE_TYPE=cso PORO_CASE_ID=cso_Rsin1e5_10cycles PORO_R_SIN=1e5 \
  singularity exec ../Singularity_fenics2019.img \
  mpirun.mpich -launcher fork -np 6 python3 -u run_lv_case.py

env PORO_CASE_TYPE=cso PORO_CASE_ID=cso_Rsin1e6_10cycles PORO_R_SIN=1e6 \
  singularity exec ../Singularity_fenics2019.img \
  mpirun.mpich -launcher fork -np 6 python3 -u run_lv_case.py
```

Use different scheduler jobs when running the four cases concurrently. Each
case writes only to `outputs/$PORO_CASE_ID`.

## Post-processing cycle 10

Waveform plots can be produced with the container Python interpreter. Darcy
surface integration and VTK field processing require a Python interpreter with
VTK, such as `pvpython` from ParaView.

```bash
singularity exec ../Singularity_fenics2019.img \
  python3 postprocess.py \
  --baseline-case outputs/baseline_Rsin100_10cycles \
  --cso-case 1e4=outputs/cso_Rsin1e4_10cycles \
  --cso-case 1e5=outputs/cso_Rsin1e5_10cycles \
  --cso-case 1e6=outputs/cso_Rsin1e6_10cycles \
  --experiment-csv coronary_inflow_waveform.csv \
  --cycle 10
```

If VTK is supplied by a separate ParaView installation, run:

```bash
pvpython postprocess.py \
  --baseline-case outputs/baseline_Rsin100_10cycles \
  --cso-case 1e4=outputs/cso_Rsin1e4_10cycles \
  --cso-case 1e5=outputs/cso_Rsin1e5_10cycles \
  --cso-case 1e6=outputs/cso_Rsin1e6_10cycles \
  --experiment-csv coronary_inflow_waveform.csv \
  --cycle 10
```

The cycle argument is human-readable. `--cycle 10` selects zero-based solver
cycle 9, corresponding to global simulation time 5940–6600 ms.

Generated products are written to:

```text
outputs/cycle10_baseline_figure3_style_v1/
outputs/cycle10_P50_35_midwall_paper_postprocessing_v1/
```

To generate waveform products on a system without VTK, add `--skip-fields`.

## Important parameters

The public production configuration uses:

```text
BCL                         660 ms
time step                   1 ms
completed cardiac cycles    10
C_sin                       0.05
P50                         35 mmHg
kP                          5 mmHg
minimum conductance         0.42
pressure jump gamma0        1e-3
normal flux penalty         1e3
MPI ranks                   6
```

Do not reuse an existing case ID unless its previous output has been moved to
a different directory.

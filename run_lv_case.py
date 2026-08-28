"""
LV-only poroelastic closed-loop driver for the minimal lvporo_min repo.

Run from this repo root:
    python3 run_lv_case.py

or with MPI:
    mpirun.mpich -np 4 python3 run_lv_case.py

This file was adapted from LVelectromechanics_pctrl.py. It removes the old
heArt_py3 sys.path imports and postprocessing imports, and calls the local
minimal LV-only solver package instead.
"""

import os
from dolfin import *

from lvporo_min.closedloop.run_BiV_ClosedLoop_pctrl_lvporo import run_BiV_ClosedLoop


# -----------------------------------------------------------------------------
# I/O settings
# -----------------------------------------------------------------------------
# Run from lv_poro_min_repo.  The two public mesh files are expected in the
# sibling ../LVMesh directory.
IODetails = {
    "casename": "ellipsoidal_baselinegeo",
    "directory_me": "../LVMesh/",
    "directory_ep": "../LVMesh/",
    "outputfolder": "./outputs/",
    "folderName": "",
    "caseID": "baseline_Rsin100_10cycles",
    "isLV": True,
}


# -----------------------------------------------------------------------------
# Material and circulation parameters
# -----------------------------------------------------------------------------
contRactility = 1200e3  # 600e3
endo_contRactility = contRactility
mid_contRactility = contRactility
epi_contRactility = contRactility

GuccioneParams = {
    "ParamsSpecified": True,
    "Passive model": {"Name": "Guccione"},
    "Passive params": {
        "Cparam": Constant(130.0),
        "bff": Constant(29.0),
        "bfx": Constant(13.3),
        "bxx": Constant(26.6),
    },
    "Active model": {
        "Name": "Time-varying",
        "ischemia": False,
    },
    "Active params": {
        "tau": 25,
        "t_trans": 300,
        "B": 4.75,
        "t0": 275,
        "l0": 1.58,
        "Tmax": Constant(contRactility),
        "Ca0": 4.35,
        "Ca0max": 4.35,
        "lr": 1.85,
    },
    "HomogenousActivation": True,
    "deg": 4,
    "Kappa": 1e5,
    "incompressible": True,
}

Circparam = {
    "Ees_la": 10,
    "A_la": 2.67,
    "B_la": 0.019,
    "V0_la": 10,
    "Tmax_la": 120,
    "tau_la": 25,
    "tdelay_la": 160,
    "Csa": 0.01,  # original: 0.0032
    "Cad": 0.013,
    "Csv": 0.12,  # 0.28
    "Vsa0": 360,
    "Vsv0": 3370.0,
    "Vad0": 40,
    "Rav": 2000.0,
    "Rsv": 100.0,
    "Rsa": 18000,  # original: 18000, 58000
    "Rad": 106000,  # original: 21200
    "Rmv": 2000.0,
    # volumes
    "V_sv": 3580,  # 3700
    "V_LV": 97,  # 112
    "V_sa": 450,  # 740
    "V_ad": 147,  # 100
    "V_LA": 222,  # 12
    # coronary / perfusion compartment
    "V_per": 35,
    "V_per0": 32,
    "R_per": 4800000,
    "C_per": 0.03,
    # coronary sinus
    "V_sin": 250,
    "V_sin0": 190,
    "R_sin": 100,
    "C_sin": 0.05,#0.18,
    # Code cycles are zero based: stop_iter=9 writes ten complete cycles 0--9.
    "stop_iter": 9,
}

# Hemodynamically equilibrated state used by the paper production runs.  It is
# embedded here so the public reproduction does not depend on a private output
# directory or a separate warm-start wrapper.
HEMODYNAMIC_WARM_START = {
    "V_sa": 439.0024763384681,
    "V_ad": 136.72263870280628,
    "V_sv": 3552.4659037913802,
    "V_LA": 193.71849282039437,
    "V_per": 71.12117942175881,
    "V_sin": 267.58267467890107,
}
Circparam.update(HEMODYNAMIC_WARM_START)


# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------
SimDetails = {
    # Minimal repo switches: LV-only + poroelastic + pressure-control.
    "isLV": True,
    "isBiV": False,
    "isFCH": False,
    "iswaorta": False,
    "ispctrl": True,
    "poro": True,

    "diaplacementInfo_ref": False,
    "HeartBeatLength": 660.0,
    "dt": 1.0,
    "writeStep": 40.0,
    "GiccioneParams": GuccioneParams,
    "nLoadSteps": 15,
    "DTI_EP": False,
    "DTI_ME": False,
    "d_iso": 1.5 * 0.005,
    "d_ani_factor": 4.0,
    "ploc": [[1.4, 1.4, -3.0, 2.0, 1]],
    "pacing_timing": [[4.0, 20.0]],
    "Isclosed": True,
    "closedloopparam": Circparam,
    "Ischemia": False,
    "springbc": 0,
    # Berger-type three-field element: CG1(u)-CG1(W)-DG0(p).
    "Mechanics Discretization": "P1P1P0_Berger",
    "topid": 4,
    "LVendoid": 2,
    "RVendoid": 0,
    "epiid": 1,
    "abs_tol": 1e-8,
    "rel_tol": 1e-9,
    "isunloading": False,
    "isunloadingonly": False,
    "epiid_Kadj_coeff": [50, 10],
    "permeability": 7.0e-5,  # reference permeability k0
    # Berger local pressure-jump stabilization. gamma = gamma0 / pressure_ref.
    # Treat gamma0 below as an INITIAL value; sweep 1e0...1e-5 and select the
    # smallest value that removes checkerboarding without altering mean pressure/flow.
    "pressure_jump_gamma0": 1.0e-3,
    "pressure_ref": 100.0 / 0.0075,  # 100 mmHg in the code pressure unit (Pa)
    # Normal-only impermeability penalty for the CG1 nominal flux on myocardial
    # boundaries. Berger 2017 uses a boundary Lagrange multiplier instead.
    "normal_flux_penalty": 1.0e3,
    "flux_boundary_ids": [1, 2, 4],  # epicardium, LV endocardium, basal plane
    "p_a": 0.0,
    "p_v": 0.0,
    "beta_a": 7.0e-8,
    "beta_v": 0.8e-5,
    # Smooth LV-pressure-dependent reduction of arterial tissue conductance.
    # Keep the legacy equations as the default; enable explicitly with
    # PORO_CORONARY_COMPRESSION_MODEL=plv_sigmoid.
    "coronary_compression_model": "plv_sigmoid",
    "coronary_g_min": 0.42,
    "coronary_P50_mmHg": 35.0,
    "coronary_kP_mmHg": 5.0,
    # Write compact reviewer-facing diagnostics and skip expensive field
    # projections when PORO_FAST_DIAGNOSTICS=1.
    "fast_diagnostics": True,
    # Write the reviewer-requested primary three-field solution as PVD series.
    "production_field_output": True,
    # Write the time-independent harmonic transmural coordinate and normal.
    # These fields are diagnostics only and do not enter the nonlinear problem.
    "write_transmural_fields": True,
    # Optional read-only regional mass/source/sink diagnostics for the
    # subendocardial, mid-wall, and subepicardial reference thirds.
    "write_regional_transmural_diagnostics": True,
    "regional_diagnostics_write_step": 1,
    # ischemia / regional contractility hooks retained for compatibility
    "Tmax_endo": Constant(endo_contRactility),
    "Tmax_mid": Constant(mid_contRactility),
    "Tmax_epi": Constant(epi_contRactility),
}


def _apply_environment_overrides():
    """Allow reproducible parameter studies without editing this case file."""
    scalar_overrides = {
        "PORO_DT": (SimDetails, "dt", float),
        "PORO_BCL": (SimDetails, "HeartBeatLength", float),
        "PORO_GAMMA0": (SimDetails, "pressure_jump_gamma0", float),
        "PORO_FLUX_PENALTY": (SimDetails, "normal_flux_penalty", float),
        "PORO_BETA_V": (SimDetails, "beta_v", float),
        "PORO_CORONARY_G_MIN": (SimDetails, "coronary_g_min", float),
        "PORO_CORONARY_P50_MMHG": (SimDetails, "coronary_P50_mmHg", float),
        "PORO_CORONARY_KP_MMHG": (SimDetails, "coronary_kP_mmHg", float),
        "PORO_WRITE_STEP": (SimDetails, "writeStep", float),
        "PORO_REGIONAL_DIAGNOSTICS_WRITE_STEP": (
            SimDetails, "regional_diagnostics_write_step", int
        ),
        "PORO_STOP_CYCLE": (Circparam, "stop_iter", int),
        "PORO_R_SIN": (Circparam, "R_sin", float),
        "PORO_C_SIN": (Circparam, "C_sin", float),
        "PORO_MAX_TIME_MS": (SimDetails, "max_time_ms", float),
    }
    for env_name, (mapping, key, converter) in scalar_overrides.items():
        if env_name in os.environ:
            mapping[key] = converter(os.environ[env_name])
    if "PORO_CASE_ID" in os.environ:
        IODetails["caseID"] = os.environ["PORO_CASE_ID"]
    if "PORO_CORONARY_COMPRESSION_MODEL" in os.environ:
        SimDetails["coronary_compression_model"] = \
            os.environ["PORO_CORONARY_COMPRESSION_MODEL"].strip().lower()
    if "PORO_FAST_DIAGNOSTICS" in os.environ:
        SimDetails["fast_diagnostics"] = os.environ["PORO_FAST_DIAGNOSTICS"].lower() \
            not in ("0", "false", "no")
    if "PORO_WRITE_PRODUCTION_FIELDS" in os.environ:
        SimDetails["production_field_output"] = \
            os.environ["PORO_WRITE_PRODUCTION_FIELDS"].lower() \
            not in ("0", "false", "no")
    if "PORO_WRITE_TRANSMURAL_FIELDS" in os.environ:
        SimDetails["write_transmural_fields"] = \
            os.environ["PORO_WRITE_TRANSMURAL_FIELDS"].lower() \
            not in ("0", "false", "no")
    if "PORO_WRITE_REGIONAL_TRANSMURAL_DIAGNOSTICS" in os.environ:
        SimDetails["write_regional_transmural_diagnostics"] = \
            os.environ["PORO_WRITE_REGIONAL_TRANSMURAL_DIAGNOSTICS"].lower() \
            not in ("0", "false", "no")

    case_type = os.environ.get("PORO_CASE_TYPE", "baseline").strip().lower()
    if case_type not in ("baseline", "cso"):
        raise ValueError("PORO_CASE_TYPE must be 'baseline' or 'cso'")
    if "PORO_CASE_ID" not in os.environ:
        resistance = Circparam["R_sin"]
        resistance_tag = "{:g}".format(resistance).replace("+", "")
        IODetails["caseID"] = "{}_Rsin{}_10cycles".format(
            case_type, resistance_tag)

    SimDetails["hemodynamic_warm_start_source"] = "embedded_public_warm_start"
    SimDetails["hemodynamic_warm_start_time_ms"] = 0.0
    SimDetails["hemodynamic_warm_start_state"] = dict(HEMODYNAMIC_WARM_START)


def _prepare_output_dir():
    """Create the output directory expected by exportfiles before the solver opens files."""
    outdir = os.path.join(
        IODetails["outputfolder"],
        IODetails["folderName"] + IODetails["caseID"],
    )
    if MPI.rank(MPI.comm_world) == 0:
        os.makedirs(outdir, exist_ok=True)
    MPI.barrier(MPI.comm_world)


def _print_mesh_reminder():
    if MPI.rank(MPI.comm_world) != 0:
        return
    me_file = os.path.join(IODetails["directory_me"], IODetails["casename"] + ".hdf5")
    ep_file = os.path.join(IODetails["directory_ep"], IODetails["casename"] + "_refine.hdf5")
    print("Expected mechanics mesh:", me_file, flush=True)
    print("Expected EP mesh       :", ep_file, flush=True)
    if not os.path.exists(me_file):
        print("WARNING: mechanics mesh file was not found at this path.", flush=True)
    if not os.path.exists(ep_file):
        print("WARNING: EP mesh file was not found at this path.", flush=True)


def main():
    _apply_environment_overrides()
    _prepare_output_dir()
    _print_mesh_reminder()
    if MPI.rank(MPI.comm_world) == 0:
        print("Case type            : {}".format(
            os.environ.get("PORO_CASE_TYPE", "baseline")), flush=True)
        print("R_sin                : {}".format(Circparam["R_sin"]), flush=True)
        print("Completed cycles     : 10 (code cycles 0--9)", flush=True)
        print("Embedded warm start  : {}".format(
            HEMODYNAMIC_WARM_START), flush=True)
    run_BiV_ClosedLoop(IODet=IODetails, SimDet=SimDetails)


if __name__ == "__main__":
    main()

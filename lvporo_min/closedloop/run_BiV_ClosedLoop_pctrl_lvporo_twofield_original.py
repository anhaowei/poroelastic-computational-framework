import sys, math
import os as os
import numpy as np
from mpi4py import MPI as pyMPI

import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)


from dolfin import *
import dolfin as dolfin

# from fenicstools import *


from ..utils.oops_objects_MRC2 import printout

from ..utils.oops_objects_MRC2 import State_Variables
from ..utils.oops_objects_MRC2 import exportfiles


from ..ep.EPmodel import EPmodel

from ..mechanics.MEmodel3 import MEmodel

# from ..mechanics.MEmodel_pctrl import MEmodel
from .circ import CLmodel  # LV-only closed-loop model

# from ..mechanics.volume_ca import MeshModifier




def run_BiV_ClosedLoop(IODet, SimDet):
    if "fiber_fspace_deg" in SimDet: 
        deg = SimDet["fiber_fspace_deg"]
    else:
        deg = 4
    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["quadrature_degree"] = deg

    casename = IODet["casename"]
    directory_me = IODet["directory_me"]
    directory_ep = IODet["directory_ep"]
    outputfolder = IODet["outputfolder"]
    folderName = IODet["folderName"] + IODet["caseID"] + "/"
    if "isLV" in list(SimDet.keys()):
        isLV = SimDet["isLV"]
    else:
        isLV = False  # Default
    if "iswaorta" in list(SimDet.keys()):
        iswaorta = SimDet["iswaorta"]
    else:
        iswaorta = False  # Default
    if "isFCH" in list(SimDet.keys()):
        isFCH = SimDet["isFCH"]
    else:
        isFCH = False  # Default
    if "isBiV" in list(SimDet.keys()):
        isBiV = SimDet["isBiV"]
    else:
        isBiV = False  # Default

    delTat = SimDet["dt"]

    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Read EP data from HDF5 Files
    mesh_ep = Mesh()
    comm_common = mesh_ep.mpi_comm()

    meshfilename_ep = directory_ep + casename + "_refine.hdf5"
    f = HDF5File(comm_common, meshfilename_ep, "r")
    f.read(mesh_ep, casename, False)

    File(outputfolder + folderName + "mesh_ep.pvd") << mesh_ep

    facetboundaries_ep = MeshFunction("size_t", mesh_ep, 2)
    f.read(facetboundaries_ep, casename + "/" + "facetboundaries")

    matid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())
    AHAid_ep = MeshFunction("size_t", mesh_ep, mesh_ep.topology().dim())

    if f.has_dataset(casename + "/" + "matid"):
        f.read(matid_ep, casename + "/" + "matid")
    else:
        matid_ep.set_all(0)

    if f.has_dataset(casename + "/" + "AHAid"):
        f.read(AHAid_ep, casename + "/" + "AHAid")
    else:
        AHAid_ep.set_all(0)

    deg_ep = 4

    Quadelem_ep = FiniteElement(
        "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
    )
    Quadelem_ep._quad_scheme = "default"
    Quad_ep = FunctionSpace(mesh_ep, Quadelem_ep)


    if "fiber_fspace" in list(SimDet.keys()) and "fiber_fspace_deg" in list(SimDet.keys()):
        VQuadelem_ep = VectorElement(
            SimDet["fiber_fspace"], mesh_ep.ufl_cell(), degree=SimDet["fiber_fspace_deg"], quad_scheme="default"
        )
        VQuadelem_ep._quad_scheme = "default"
    else:
        VQuadelem_ep = VectorElement(
            "Quadrature", mesh_ep.ufl_cell(), degree=deg_ep, quad_scheme="default"
        )
        VQuadelem_ep._quad_scheme = "default"




    fiberFS_ep = FunctionSpace(mesh_ep, VQuadelem_ep)

    f0_ep = Function(fiberFS_ep)
    s0_ep = Function(fiberFS_ep)
    n0_ep = Function(fiberFS_ep)

    if SimDet["DTI_EP"] is True:
        f.read(f0_ep, casename + "/" + "eF_DTI")
        f.read(s0_ep, casename + "/" + "eS_DTI")
        f.read(n0_ep, casename + "/" + "eN_DTI")
    else:
        f.read(f0_ep, casename + "/" + "eF")
        f.read(s0_ep, casename + "/" + "eS")
        f.read(n0_ep, casename + "/" + "eN")

    f.close()

    comm_ep = mesh_ep.mpi_comm()

    # Define state variables
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    state_obj = State_Variables(comm_ep, SimDet)
    state_obj.dt.dt = delTat
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    EPparams = {
        "EPmesh": mesh_ep,
        "deg": 4,
        "matid": matid_ep,
        "facetboundaries": facetboundaries_ep,
        "f0": f0_ep,
        "s0": s0_ep,
        "n0": n0_ep,
        "state_obj": state_obj,
        "d_iso": SimDet["d_iso"],
        "d_ani_factor": SimDet["d_ani_factor"],
        "AHAid": AHAid_ep,
        "matid": matid_ep,
    }

    if "ploc" in list(SimDet.keys()):
        EPparams.update({"ploc": SimDet["ploc"]})
    if "Ischemia" in list(SimDet.keys()):
        EPparams.update({"Ischemia": SimDet["Ischemia"]})
    if "pacing_timing" in list(SimDet.keys()):
        EPparams.update({"pacing_timing": SimDet["pacing_timing"]})

    # Define EP model and solver
    EPmodel_ = EPmodel(EPparams)
    EpiBCid_ep = EPmodel_.MarkStimulus()

    solver_FHN = EPmodel_.Solver()
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Mechanics Mesh

    mesh_me = Mesh()
    mesh_me_params = {
        "directory": directory_me,
        "casename": casename,
        "fibre_quad_degree": 4,
        "outputfolder": outputfolder,
        "foldername": folderName,
        "state_obj": state_obj,
        "common_communicator": comm_common,
        "MEmesh": mesh_me,
        "isLV": isLV,
    }

    MEmodel_ = MEmodel(mesh_me_params, SimDet)
    solver_elas = MEmodel_.Solver()
    comm_me = MEmodel_.mesh_me.mpi_comm()

    # Set up export class
    export = exportfiles(comm_me, comm_ep, IODet, SimDet)

    export.exportVTKobj("facetboundaries_ep.pvd", facetboundaries_ep)
    export.exportVTKobj("EpiBCid_ep.pvd", EpiBCid_ep)
    # export.exportVTKobj("f0.pvd", project(MEmodel_.Mesh.f0, VectorFunctionSpace(MEmodel_.Mesh.mesh, "DG", 0)))

    F_ED = Function(MEmodel_.TF)

    if "AHA_segments" in list(SimDet.keys()):
        AHA_segments = SimDet["AHA_segments"]
    else:
        AHA_segments = [0]

    # Get Unloaded volumes
    V_LV_unload = MEmodel_.GetLVV()
    V_RV_unload = MEmodel_.GetRVV()

    nloadstep = SimDet["nLoadSteps"]

    # Unloading LV to get new reference geometry
    MEmodel_.LVCavityvol.vol = MEmodel_.GetLVV()
    MEmodel_.LVCavitypres.pres = 0.0
    MEmodel_.RVCavitypres.pres = 0.0

    # export.writePV(MEmodel_, 0);
    export.hdf.write(MEmodel_.mesh_me, "ME/mesh")
    export.hdf.write(EPmodel_.mesh_ep, "EP/mesh")

    default_params = {
        "EDP": 12.0,
        "maxit": 20,
        "restol": 1e-3,
        "drestol": 1e-4,
        "EDPtol": 1e-1,
        "preinc": 1,
        "LVangle": [60, -60],
    }
    # default_params.update(params)

    EDP = default_params["EDP"]
    LVangle = default_params["LVangle"]
    maxit = default_params["maxit"]
    restol = default_params["restol"]
    drestol = default_params["drestol"]
    EDPtol = default_params["EDPtol"]
    preinc = default_params["preinc"]

    it = 0
    #tempfile  = File(outputfolder + folderName + "displacement.pvd")
    tempfile  = File(outputfolder + folderName + "mass.pvd")
    tempfile2 = File(outputfolder + folderName + "massgrad.pvd")
    tempfile3 = File(outputfolder + folderName + "DarcyFlux.pvd")
    tempfile4 = File(outputfolder + folderName + "DarcyNormL.pvd")
    tempfile5 = File(outputfolder + folderName + "pressure.pvd")
    DarcyV0 = FunctionSpace(MEmodel_.mesh_me, "CG", 1)
    Ttm, ntm = MEmodel_.GetNormaldirtction()
    #tempfile6 = File(outputfolder + folderName + "Laplace.pvd")
    #tempfile6 << ntm
    while 1:
        printout("Loading", comm_me)
        MEmodel_.LVCavitypres.pres += (EDP / 0.0075) / nloadstep
        if isBiV or isFCH:
            MEmodel_.RVCavitypres.pres += (EDP / 0.0075) / 2 / nloadstep

        solver_elas.solvenonlinear()

        if(it % 10 == 0):
            qdarcy = MEmodel_.GetDarcy()
            q_darcyfunc = project(inner(qdarcy,ntm),DarcyV0)
            q_darcyfunc.rename("n_tm", "n_tm")
            #tempfile << MEmodel_.GetDisplacement()
            tempfile << MEmodel_.GetMass()
            tempfile2 << MEmodel_.GetMassGrad()
            # tempfile3 << qdarcy 
            tempfile4 << q_darcyfunc
            tempfile5 << MEmodel_.GetPorePressure()

        export.writePV(MEmodel_, 0)
        export.hdf.write(MEmodel_.GetDisplacement(), "ME/u_loading", it)
        it += 1

        # F_ED.vector()[:] = (
        #    project(
        #        MEmodel_.GetFmat(),
        #        MEmodel_.TF,
        #        solver_type="mumps",
        #        form_compiler_parameters={"representation": "quadrature"},
        #    )
        #    .vector()
        #    .get_local()[:]
        # )

        printout(
            "LV Pressure = "
            + str(MEmodel_.GetLVP() * 0.0075)
            + " LV Vol = "
            + str(MEmodel_.GetLVV())  # GetVolumeComputation()),
            + "RV Pressure = "
            + str(MEmodel_.GetRVP() * 0.0075)
            + "RV Vol = "
            + str(MEmodel_.GetRVV()),  # GetVolumeComputation()),
            comm_me
        )

        if MEmodel_.LVCavitypres.pres * 0.0075 >= EDP:
            break

    printout("volume = " + str(MEmodel_.GetLVV()), comm_me)
    # return
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
    # Declare communicator based on mpi4py
    # eCC, eRR, eLL, deformedMesh, deformedBoundary = MEmodel_.GetDeformedBasis({})

    # fStrain = MEmodel_.GetFiberstrain(F_ED)
    fStrain_uL = MEmodel_.GetFiberstrainUL()
    #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

    # Closed-loop phase
    stop_iter = SimDet["closedloopparam"]["stop_iter"]

    # Systemic circulation

    # Pulmonary circulation
    heart_shape = [isLV, iswaorta, isFCH]
    if all(not x for x in heart_shape):
        # if not isLV and not iswaorta and not isFCH:
        Cpa = SimDet["closedloopparam"]["Cpa"]
        Cpv = SimDet["closedloopparam"]["Cpv"]
        Vpa0 = SimDet["closedloopparam"]["Vpa0"]
        Vpv0 = SimDet["closedloopparam"]["Vpv0"]
        Rpv = SimDet["closedloopparam"]["Rpv"]
        Rtv = SimDet["closedloopparam"]["Rtv"]
        Rpa = SimDet["closedloopparam"]["Rpa"]
        Rpvv = SimDet["closedloopparam"]["Rpvv"]
        V_pv = SimDet["closedloopparam"]["V_pv"]
        V_pa = SimDet["closedloopparam"]["V_pa"]
        V_RA = SimDet["closedloopparam"]["V_RA"]

    isrestart = 0
    prev_cycle = 0
    cnt = 0

    Qtv = 0
    Qpa = 0
    Qpv = 0
    Qpvv = 0
    Qlvad = 0
    Qlara = 0

    if "Q_tv" in list(SimDet["closedloopparam"].keys()):
        Qtv = SimDet["closedloopparam"]["Q_tv"]
    if "Q_pa" in list(SimDet["closedloopparam"].keys()):
        Qpa = SimDet["closedloopparam"]["Q_pa"]
    if "Q_pv" in list(SimDet["closedloopparam"].keys()):
        Qpv = SimDet["closedloopparam"]["Q_pv"]
    if "Q_pvv" in list(SimDet["closedloopparam"].keys()):
        Qpvv = SimDet["closedloopparam"]["Q_pvv"]
    if "Q_lvad" in list(SimDet["closedloopparam"].keys()):
        Qlvad = SimDet["closedloopparam"]["Q_lvad"]
    if "Q_lara" in list(SimDet["closedloopparam"].keys()):
        Qlara = SimDet["closedloopparam"]["Q_lara"]

    # Parameters for LVAD #############################################
    LVADrpm = 0
    LVADscale = 0
    if "Q_lvad_rpm" in list(SimDet["closedloopparam"].keys()):
        LVADrpm = SimDet["closedloopparam"]["Q_lvad_rpm"]
    if "Q_lvad_scale" in SimDet["closedloopparam"].keys():
        LVADscale = SimDet["closedloopparam"]["Q_lvad_scale"]
    if "Q_lvad_characteristic" in list(SimDet["closedloopparam"].keys()):
        QLVADFn = SimDet["closedloopparam"]["Q_lvad_characteristic"]

    Qlad = 0
    Qlcx = 0

    # Parameters for Shunt #############################################
    Shuntscale = 0.0
    Rsh = 1e9
    if "Shunt_scale" in list(SimDet["closedloopparam"].keys()):
        Shuntscale = SimDet["closedloopparam"]["Shunt_scale"]
    if "Rsh" in list(SimDet["closedloopparam"].keys()):
        Rsh = SimDet["closedloopparam"]["Rsh"]

    potential_me = Function(FunctionSpace(MEmodel_.mesh_me, "CG", 1))
    writecnt = 0

    P_LV = MEmodel_.GetLVP()  # LVCavitypres.pres
    V_LV = MEmodel_.GetLVV()  # GetVolumeComputation()
    Q_cor= MEmodel_.GetFlowRate()
    #Q_cor_n= 0 
    Q_cor_n= MEmodel_.GetFlowRate_n()
    Q_cor_endo= MEmodel_.GetFlowRateEndo()
    porepressure = MEmodel_.GetPorePressure2()
    wallvolume   = MEmodel_.GetWallVolume()
    if isBiV or isFCH:
        P_RV = MEmodel_.GetRVP()  # LVCavitypres.pres
        V_RV = MEmodel_.GetRVV()  # GetVolumeComputation()

    if isLV or iswaorta:
        CLmodel_ = CLmodel(SimDet, V_LV)
    elif isBiV or isFCH:
        CLmodel_ = CLmodel_biv(SimDet, V_LV, V_RV)

    it_ = 0

    while 1:
        if state_obj.cycle > stop_iter:
            break
        # if state_obj.t > 100:
        # break

        if isLV or iswaorta:
            Q_cor= MEmodel_.GetFlowRate()
            Q_cor_n= MEmodel_.GetFlowRate_n()
            Q_cor_endo= MEmodel_.GetFlowRateEndo()
            porepressure = MEmodel_.GetPorePressure2()
            wallvolume   = MEmodel_.GetWallVolume()
            params = {
                "P_LV": P_LV,
                "V_LV": V_LV,
                "t": state_obj.t,
                "delTat": state_obj.dt.dt,
                "DarcyFlowRate": Q_cor,
                "DarcyFlowRate_n": Q_cor_n,
                "porepressure":porepressure,
                "wallvolume":wallvolume,
            }
        elif isBiV or isFCH:
            params = {
                "P_LV": P_LV,
                "V_LV": V_LV,
                "P_RV": P_RV,
                "V_RV": V_RV,
                "t": state_obj.t,
                "delTat": state_obj.dt.dt,
            }

        if isLV or iswaorta:
            V_LV = CLmodel_.UpdateLVV(params)
            # Q_cor= MEmodel_.GetFlowRate()
            # Q_cor_n= MEmodel_.GetFlowRate_n()
            # Q_cor_endo= MEmodel_.GetFlowRateEndo()
            # porepressure = MEmodel_.GetPorePressure2()
            # wallvolume   = MEmodel_.GetWallVolume()
        elif isBiV or isFCH:
            V_LV, V_RV = CLmodel_.UpdateLVV(params)

        if isLV or iswaorta:
            printout(
                "t = "
                + str(state_obj.t)
                + "V_LV = "
                + str(V_LV)
                + " Psa = "
                + str(CLmodel_.Psa)
                 + " Psv = "
                + str(CLmodel_.Psv)
                 + " Perfusion pressure = "
                + str(CLmodel_.Psa-CLmodel_.Psv)
                + " P_LV = "
                + str(P_LV)
                + " Q_cor= "
                + str(Q_cor)
                + " q_cor= "
                + str(Q_cor-Q_cor_n)
                + " Q_cor_n= "
                + str(Q_cor_n)
                + " Q_cor_endo="
                + str(Q_cor_endo),
               
                
                comm_me,
            )

        elif isBiV or isFCH:
            printout(
                "t = "
                + str(state_obj.t)
                + "V_LV = "
                + str(V_LV)
                + " Psa = "
                + str(CLmodel_.Psa)

                + " PLA = "
                + str(CLmodel_.GetPLoRA(params, 1))
                + " P_LV = "
                + str(P_LV)
                + " Q_cor= "
                + str(Q_cor)
                + " Q_cor_n= "
                + str(Q_cor_n),
                comm_me,
            )


        with open(outputfolder + folderName + "output_PV.txt", "a") as f_PV:
            if MPI.rank(comm_me) == 0:
                if isLV or iswaorta:
                    f_PV.write(f"{state_obj.t}, {V_LV}, {P_LV} \n")
                elif isBiV or isFCH:
                    f_PV.write(f"{state_obj.t}, {V_LV}, {P_LV}, {V_RV}, {P_RV} \n")
        with open(outputfolder + folderName + "output_qt.txt", "a") as f_q:
            if MPI.rank(comm_me) == 0:
                if isLV or iswaorta:
                    f_q.write(f"{state_obj.t}, {Q_cor-Q_cor_n} \n")
                elif isBiV or isFCH:
                    f_PV.write(f"{state_obj.t}, {V_LV}, {P_LV}, {V_RV}, {P_RV} \n")
        with open(outputfolder + folderName + "output_Qcorin.txt", "a") as f_Qin:
            if MPI.rank(comm_me) == 0:
                if isLV or iswaorta:
                    f_Qin.write(f"{state_obj.t}, {Q_cor} \n")
                elif isBiV or isFCH:
                    f_PV.write(f"{state_obj.t}, {V_LV}, {P_LV}, {V_RV}, {P_RV} \n")
        with open(outputfolder + folderName + "output_Qcorout.txt", "a") as f_Qout:
            if MPI.rank(comm_me) == 0:
                if isLV or iswaorta:
                    f_Qout.write(f"{state_obj.t}, {Q_cor_n} \n")
                elif isBiV or isFCH:
                    f_PV.write(f"{state_obj.t}, {V_LV}, {P_LV}, {V_RV}, {P_RV} \n")

        with open(outputfolder + folderName + "output_porepres.txt", "a") as f_Ppore:
            if MPI.rank(comm_me) == 0:
                if isLV or iswaorta:
                    f_Ppore.write(f"{state_obj.t}, {porepressure} \n")
                elif isBiV or isFCH:
                    f_PV.write(f"{state_obj.t}, {V_LV}, {P_LV}, {V_RV}, {P_RV} \n")

        with open(outputfolder + folderName + "output_wallvolume.txt", "a") as f_Vwall:
            if MPI.rank(comm_me) == 0:
                if isLV or iswaorta:
                    f_Vwall.write(f"{state_obj.t}, {wallvolume} \n")
                elif isBiV or isFCH:
                    f_PV.write(f"{state_obj.t}, {V_LV}, {P_LV}, {V_RV}, {P_RV} \n")


        # Newton's solver
        tol = 1e-3  # Tolerance for convergence
        max_iter = 100  # Maximum number of iteration

        def estpres(plv):
            return 1.005 * plv

        def Jf(plv):
            MEmodel_.LVCavitypres.pres = plv
            solver_elas.solvenonlinear()
            fe_v1 = MEmodel_.GetLVV()

            MEmodel_.LVCavitypres.pres = estpres(plv)
            solver_elas.solvenonlinear()
            fe_v2 = MEmodel_.GetLVV()

            return (fe_v2 - fe_v1) / (estpres(plv) - plv)

        def Jf_biv(plv, prv, lvp, lvv):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            solver_elas.solvenonlinear()

            if lvv:
                fe_v1 = MEmodel_.GetLVV()
            else:
                fe_v1 = MEmodel_.GetRVV()

            if lvp:
                MEmodel_.LVCavitypres.pres = estpres(plv)
            else:
                MEmodel_.RVCavitypres.pres = estpres(prv)
            solver_elas.solvenonlinear()

            if lvv:
                fe_v2 = MEmodel_.GetLVV()
            else:
                fe_v2 = MEmodel_.GetRVV()

            if lvp:
                return (fe_v2 - fe_v1) / (estpres(plv) - plv)
            else:
                return (fe_v2 - fe_v1) / (estpres(prv) - prv)

        def run_plvr(plv, prv):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            solver_elas.solvenonlinear()

            return MEmodel_.GetLVV(), MEmodel_.GetRVV()

        # vlv, vrv = run_plvr(P_LV, P_RV)

        def JR_perturb(plv, prv, lvp, lvvc, rvvc):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            solver_elas.solvenonlinear()

            vlv_ = MEmodel_.GetLVV()
            vrv_ = MEmodel_.GetRVV()

            if lvp:
                MEmodel_.LVCavitypres.pres = estpres(plv)
            else:
                MEmodel_.RVCavitypres.pres = estpres(prv)
            solver_elas.solvenonlinear()

            fe_v2l = MEmodel_.GetLVV()
            fe_v2r = MEmodel_.GetRVV()

            if lvp:
                return (
                    (fe_v2l - vlv_) / (estpres(plv) - plv),
                    (fe_v2r - vrv_) / (estpres(plv) - plv),
                    vlv_ - lvvc,
                )
            else:
                return (
                    (fe_v2l - vlv_) / (estpres(prv) - prv),
                    (fe_v2r - vrv_) / (estpres(prv) - prv),
                    vrv_ - rvvc,
                )

        # ax_l, by_l, cz_l = JR_perturb(P_LV, P_RV, 1, V_LV, V_RV)
        # ax_r, by_r, cz_r = JR_perturb(P_LV, P_RV, 0, V_LV, V_RV)

        # comm_me.Barrier()

        def Rp(plv, vlv):
            MEmodel_.LVCavitypres.pres = plv
            solver_elas.solvenonlinear()
            v_t = MEmodel_.GetLVV()
            return v_t - vlv

        def Rp_biv(plv, prv, lvvc, rvvc):
            MEmodel_.LVCavitypres.pres = plv
            MEmodel_.RVCavitypres.pres = prv
            solver_elas.solvenonlinear()
            vlv = MEmodel_.GetLVV()
            vrv = MEmodel_.GetRVV()

            return vlv - lvvc, vrv - rvvc

        # Create the Newton solver
        for iter in range(max_iter):
            # Compute the residual and Jacobian
            if isLV or iswaorta:
                J = Jf(P_LV)
                F = Rp(P_LV, V_LV)

            elif isBiV or isFCH:
                # J = np.array(
                #   [
                #       [Jf_biv(P_LV, P_RV, 1, 1), Jf_biv(P_LV, P_RV, 1, 0)],
                #       [Jf_biv(P_LV, P_RV, 0, 1), Jf_biv(P_LV, P_RV, 0, 0)],
                #   ]
                # )

                # vlv, vrv = run_plvr(P_LV, P_RV)
                ax_l, by_l, cz_l = JR_perturb(P_LV, P_RV, 1, V_LV, V_RV)
                ax_r, by_r, cz_r = JR_perturb(P_LV, P_RV, 0, V_LV, V_RV)

                J = np.array(
                    [
                        [ax_l, by_l],
                        [ax_r, by_r],
                    ]
                )

                # F = Rp_biv(P_LV, P_RV, V_LV, V_RV)
                # F = np.array([[vfe] for vfe in Rp_biv(P_LV, P_RV, V_LV, V_RV)])
                F = np.array([cz_l, cz_r])

            # with open(
            #    outputfolder + folderName + "output_JRp_serial.txt", "a"
            # ) as f_JRp:
            #    if MPI.rank(comm_me) == 0:
            #        f_JRp.write(
            #            # f"t = {state_obj.t}, iter = {iter}, Rp = {F}, J = {J} \n"
            #            f"t = {state_obj.t}, iter = {iter}, J = {J}, F = {F} \n"
            #        )

            # Solve for the update
            if isLV or iswaorta:
                if abs(J) < 1e-10:
                    printout("Jac is too small: " + str(du), comm_me)
                    # if MPI.rank(comm_me) == 0:
                    # f_JRp.write(f"break due to small Jac: du = {du}.")
                    # continue
                    break
            elif isBiV or isFCH:
                if np.linalg.norm(J) < 1e-10:
                    printout("Jac is too small: " + str(du), comm_me)
                    break

            if isLV or iswaorta:
                du = -F / J
            elif isBiV or isFCH:
                du = np.dot(np.linalg.inv(J), F)

            # Update the solution
            # if abs(du) > 230:
            #     du /= 2

            if isLV or iswaorta:
                while abs(du) > 220:
                    du /= 2
            elif isBiV or isFCH:
                while np.linalg.norm(tempfiledu) > 220:
                    du /= 2

            if isLV or iswaorta:
                P_LV += du
            elif isBiV or isFCH:
                P_LV -= float(du[0])
                P_RV -= float(du[1])

            # Check for convergence
            if isLV or iswaorta:
                if abs(F) < tol and abs(du) < tol:
                    break
            elif isBiV or isFCH:
                if np.linalg.norm(F) < tol and np.linalg.norm(du) < tol:
                    break

            with open(outputfolder + folderName + "output_JRp.txt", "a") as f_JRp:
                if MPI.rank(comm_me) == 0:
                    f_JRp.write(f"t = {state_obj.t}, iter = {iter}, du = {du} \n")

            # if cnt % SimDet["writeStep"] == 0.0:
            #    export.hdf.write(MEmodel_.GetDisplacement(), "ME/u", writecnt)
            #    # export.hdf.write(c_n, "ME/u_diff", writecnt)
            #    writecnt += 1

        MEmodel_.p_a.coronary = CLmodel_.Pper
        #MEmodel_.p_v.venous = CLmodel_.Psv
        MEmodel_.p_v.venous = CLmodel_.Psin
        #Q_cor_n = MEmodel_.GetFlowRate_n() #this is using previous time step J to use J-J_n to calculate flow rate
        with open(outputfolder + folderName + "output_Psv.txt", "a") as Psv:
                if MPI.rank(comm_me) == 0:
                    Psv.write(f"{state_obj.t}, {CLmodel_.Psv} \n")
        with open(outputfolder + folderName + "output_Psa.txt", "a") as Psa:
                if MPI.rank(comm_me) == 0:
                    Psa.write(f"{state_obj.t}, {CLmodel_.Psa} \n")
        with open(outputfolder + folderName + "output_Pper.txt", "a") as Pper:
                if MPI.rank(comm_me) == 0:
                    Pper.write(f"{state_obj.t}, {CLmodel_.Pper} \n")
        with open(outputfolder + folderName + "output_Psin.txt", "a") as Psin:
                if MPI.rank(comm_me) == 0:
                    Psin.write(f"{state_obj.t}, {CLmodel_.Psin} \n")      
        

        
        with open(outputfolder + folderName + "output_Qav.txt", "a") as Qav:
                if MPI.rank(comm_me) == 0:
                    Qav.write(f"{state_obj.t}, {CLmodel_.Qav} \n")
        with open(outputfolder + folderName + "output_Qsa.txt", "a") as Qsa:
                if MPI.rank(comm_me) == 0:
                    Qsa.write(f"{state_obj.t}, {CLmodel_.Qsa} \n")
        with open(outputfolder + folderName + "output_Qad.txt", "a") as Qad:
                if MPI.rank(comm_me) == 0:
                    Qad.write(f"{state_obj.t}, {CLmodel_.Qad} \n")
        with open(outputfolder + folderName + "output_Qmv.txt", "a") as Qmv:
                if MPI.rank(comm_me) == 0:
                    Qmv.write(f"{state_obj.t}, {CLmodel_.Qmv} \n")
        with open(outputfolder + folderName + "output_Qper.txt", "a") as Qper:
                if MPI.rank(comm_me) == 0:
                    Qper.write(f"{state_obj.t}, {CLmodel_.Qper} \n")
        with open(outputfolder + folderName + "output_Qsin.txt", "a") as Qsin:
                if MPI.rank(comm_me) == 0:
                    Qsin.write(f"{state_obj.t}, {CLmodel_.Qsin} \n")
        with open(outputfolder + folderName + "output_Qsv.txt", "a") as Qsv:
                if MPI.rank(comm_me) == 0:
                    Qsv.write(f"{state_obj.t}, {CLmodel_.Qsv} \n")


        with open(outputfolder + folderName + "output_V_lv.txt", "a") as V_lv:
                if MPI.rank(comm_me) == 0:
                    V_lv.write(f"{state_obj.t}, {CLmodel_.V_LV} \n")
        with open(outputfolder + folderName + "output_V_sa.txt", "a") as V_sa:
                if MPI.rank(comm_me) == 0:
                    V_sa.write(f"{state_obj.t}, {CLmodel_.V_sa} \n")
        with open(outputfolder + folderName + "output_V_per.txt", "a") as V_per:
                if MPI.rank(comm_me) == 0:
                    V_per.write(f"{state_obj.t}, {CLmodel_.V_per} \n")
        with open(outputfolder + folderName + "output_V_ad.txt", "a") as V_ad:
                if MPI.rank(comm_me) == 0:
                    V_ad.write(f"{state_obj.t}, {CLmodel_.V_ad} \n")
        with open(outputfolder + folderName + "output_V_sin.txt", "a") as V_sin:
                if MPI.rank(comm_me) == 0:
                    V_sin.write(f"{state_obj.t}, {CLmodel_.V_sin} \n")
        with open(outputfolder + folderName + "output_V_LA.txt", "a") as V_LA:
                if MPI.rank(comm_me) == 0:
                    V_LA.write(f"{state_obj.t}, {CLmodel_.V_LA} \n")
        with open(outputfolder + folderName + "output_V_sv.txt", "a") as V_sv:
                if MPI.rank(comm_me) == 0:
                    V_sv.write(f"{state_obj.t}, {CLmodel_.V_sv} \n") 
        

       

        if(cnt % 10 == 0):
            qdarcy = MEmodel_.GetDarcy()
            q_darcyfunc = project(inner(qdarcy,ntm),DarcyV0)
            q_darcyfunc.rename("n_tm", "n_tm")
            #tempfile << MEmodel_.GetDisplacement()
            tempfile << MEmodel_.GetMass()
            tempfile2 << MEmodel_.GetMassGrad()
            # tempfile3 << qdarcy 
            tempfile4 << q_darcyfunc
            tempfile5 << MEmodel_.GetPorePressure()
            # #tempfile << MEmodel_.GetDisplacement() #LCL
            # tempfile << MEmodel_.GetMass()
            # tempfile2 << MEmodel_.GetMassGrad()
            # tempfile3 << MEmodel_.GetDarcy()
            # tempfile4 << MEmodel_.GetNormaldirtctionDarcy()
            # tempfile5 << MEmodel_.GetPorePressure()

        state_obj.tstep = state_obj.tstep + state_obj.dt.dt
        state_obj.cycle = math.floor(state_obj.tstep / state_obj.BCL)
        state_obj.t = state_obj.tstep - state_obj.cycle * state_obj.BCL

        MEmodel_.t_a.vector()[:] = state_obj.t

        isrestart = 0
        state_obj.dt.dt = delTat
        if state_obj.t >= 400.0:
            state_obj.dt.dt = 2.0 * delTat

        # Reset phi and r in EP at end of diastole
        if state_obj.t < state_obj.dt.dt:
            EPmodel_.reset()

        printout("Solving FHN", comm_me)
        solver_FHN.solvenonlinear()
        if isrestart == 0:
            MEmodel_.UpdateVar()  # For damping
            EPmodel_.UpdateVar()

        # Interpolate phi to mechanics mesh
        potential_ref = EPmodel_.interpolate_potential_ep2me_phi(
            V_me=Function(FunctionSpace(MEmodel_.mesh_me, "CG", 1))
        )
        potential_ref.rename("v_ref", "v_ref")

        potential_me.vector()[:] = potential_ref.vector().get_local()[:]

        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
        if MPI.rank(comm_ep) == 0:
            print("UPdating isActiveField and tInitiationField")

        MEmodel_.activeforms.update_activationTime(
            potential_n=potential_me, comm=comm_me
        )

        F_n = MEmodel_.GetFmat()
        fstress_DG = project(
            MEmodel_.Getfstress(),
            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
            form_compiler_parameters={"representation": "uflacs"},
        )
        fstress_DG.rename("fstress", "fstress")

        if "probepts" in list(SimDet.keys()):
            probesfstress = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesfstress(fstress_DG)

        Eul_fiber_BiV_DG = project(
            fStrain_uL,
            FunctionSpace(MEmodel_.mesh_me, "DG", 0),
            form_compiler_parameters={"representation": "uflacs"},
        )

        Eul_fiber_BiV_DG.rename("Eff", "Eff")
        if "probepts" in list(SimDet.keys()):
            # x = np.array(SimDet["probepts"])
            probesEul_fiber = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesEul_fiber(Eul_fiber_BiV_DG)

            probesE_circ_BiV = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesE_long_BiV = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )
            probesE_radi_BiV = Probes(
                x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1)
            )

        # postprocess and write
        #  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -

        ## ----------------- Compute Natural Strain -----------------------------------------------------------------------------
        # E_circ_BiV, E_circ_BiV_ = MEmodel_.GetFiberNaturalStrain(
        #    F_ED, eCC, AHA_segments
        # )
        # E_long_BiV, E_long_BiV_ = MEmodel_.GetFiberNaturalStrain(
        #    F_ED, eLL, AHA_segments
        # )
        # E_radi_BiV, E_radi_BiV_ = MEmodel_.GetFiberNaturalStrain(
        #    F_ED, eRR, AHA_segments
        # )
        ## --------------------------------------------------------------------------------------------------------------------
        #
        # E_circ_BiV_DG = project(
        #    E_circ_BiV_,
        #    FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #    form_compiler_parameters={"representation": "uflacs"},
        # )
        # E_circ_BiV_DG.rename("Ecc", "Ecc")
        # if "probepts" in list(SimDet.keys()):
        #    probesE_circ_BiV(E_circ_BiV_DG)

        # E_long_BiV_DG = project(
        #    E_long_BiV_,
        #    FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #    form_compiler_parameters={"representation": "uflacs"},
        # )
        # E_long_BiV_DG.rename("Ell", "Ell")
        # if "probepts" in list(SimDet.keys()):
        #    probesE_long_BiV(E_long_BiV_DG)

        # E_radi_BiV_DG = project(
        #    E_radi_BiV_,
        #    FunctionSpace(MEmodel_.mesh_me, "DG", 0),
        #    form_compiler_parameters={"representation": "uflacs"},
        # )
        # E_radi_BiV_DG.rename("Err", "Err")
        # if "probepts" in list(SimDet.keys()):
        #    probesE_radi_BiV(E_radi_BiV_DG)

        # Compute IMP
        imp = project(
            MEmodel_.GetIMP(),
            FunctionSpace(MEmodel_.mesh_me, "DG", 1),
            form_compiler_parameters={"representation": "uflacs"},
        )
        imp.rename("imp", "imp")

        imp2 = project(
            MEmodel_.GetIMP2(),
            FunctionSpace(MEmodel_.mesh_me, "DG", 1),
            form_compiler_parameters={"representation": "uflacs"},
        )
        imp2.rename("imp2", "imp2")

        if "probepts" in list(SimDet.keys()):
            x = np.array(SimDet["probepts"])
            probesIMP = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))
            probesIMP(imp)

            probesIMP2 = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "DG", 1))
            probesIMP2(imp2)

            probesIMP3 = Probes(x.flatten(), FunctionSpace(MEmodel_.mesh_me, "CG", 1))
            probesIMP3(MEmodel_.GetP())

            # broadcast from proc 0 to other processes
            rank = comm_me_.Get_rank()
            a = probesIMP3.array()  ## probe will only send to rank =0
            if not rank == 0:
                a = np.empty(len(x))

            comm_me_.Bcast(a, root=0)

        export.writePV(MEmodel_, state_obj.tstep)

        if cnt % SimDet["writeStep"] == 0.0:
            export.writetpt(MEmodel_, state_obj.tstep)
            export.hdf.write(MEmodel_.GetDisplacement(), "ME/u", writecnt)
            export.hdf.write(potential_ref, "ME/potential_ref", writecnt)
            # export.hdf.write(E_circ_BiV_DG, "ME/Ecc", writecnt)
            # export.hdf.write(E_long_BiV_DG, "ME/Ell", writecnt)
            # export.hdf.write(E_radi_BiV_DG, "ME/Err", writecnt)
            export.hdf.write(Eul_fiber_BiV_DG, "ME/Eff", writecnt)
            export.hdf.write(fstress_DG, "ME/fstress", writecnt)
            export.hdf.write(imp, "ME/imp", writecnt)
            export.hdf.write(imp2, "ME/imp2", writecnt)
            export.hdf.write(MEmodel_.GetP(), "ME/imp_constraint", writecnt)

            export.hdf.write(EPmodel_.getphivar(), "EP/phi", writecnt)
            export.hdf.write(EPmodel_.getrvar(), "EP/r", writecnt)
            export.hdf.write(potential_ref, "EP/potential_ref", writecnt)

            writecnt += 1

        if "probepts" in list(SimDet.keys()):
            fIMP = probesIMP.array()
            fIMP2 = probesIMP2.array()
            fIMP3 = probesIMP3.array()
            fStress = probesfstress.array()
            fStrain_vals = probesEul_fiber.array()
            E_circ_BiV = probesE_circ_BiV.array()
            E_long_BiV = probesE_long_BiV.array()
            E_radi_BiV = probesE_radi_BiV.array()

            export.writeIMP(MEmodel_, state_obj.tstep, fIMP)
            export.writeIMP2(MEmodel_, state_obj.tstep, fIMP2)
            export.writeIMP3(MEmodel_, state_obj.tstep, fIMP3)
            export.writefStress(MEmodel_, state_obj.tstep, fStress)
            export.writefStrain(MEmodel_, state_obj.tstep, fStrain_vals)
            export.writeCStrain(MEmodel_, state_obj.tstep, E_circ_BiV)
            export.writeLStrain(MEmodel_, state_obj.tstep, E_long_BiV)
            export.writeRStrain(MEmodel_, state_obj.tstep, E_radi_BiV)

        cnt += 1


#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
if __name__ == "__main__":
    raise SystemExit(
        "Do not run this module directly. Create a case driver that imports "
        "run_BiV_ClosedLoop from lvporo_min.closedloop.run_BiV_ClosedLoop_pctrl_lvporo "
        "and passes IODet and SimDet. See run_lv_case_template.py."
    )

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
# check with open line for post processing
# save .txt out of for loop ,in the while loop

from dolfin import *
import numpy as np

from ..utils.nsolver import NSolver
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh
from .forms_MRC2 import Forms
from .activeforms_MRC2 import activeForms
from ufl import indices


class MEmodel(object):
    """
    Minimal LV-only mechanics model for the poroelastic closed-loop case.

    Design choices:
    - Supports LV only: SimDet["isLV"] must be True; BiV/FCH/aorta branches are removed.
    - Supports pressure-control mechanics: SimDet["ispctrl"] should be True.
    - Uses a poroelastic residual only: Darcy/mass balance + poroelastic momentum balance
      + active stress + rigid-body constraints.
    - Keeps the public methods used by run_BiV_ClosedLoop_pctrl_lvporo.py so the driver
      can run with only import cleanup.
    """

    def __init__(self, params, SimDet):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.SimDet = SimDet

        self.isLV = bool(SimDet.get("isLV", True))
        self.iswaorta = False
        self.isFCH = False
        self.isBiV = False
        if not self.isLV:
            raise NotImplementedError("This minimal MEmodel3.py supports LV case only.")

        self.ispctrl = bool(SimDet.get("ispctrl", True))
        if not self.ispctrl:
            raise NotImplementedError("This minimal model keeps only pressure-control LV mechanics.")

        self.islumped = bool(SimDet.get("islumped", False))
        self.deg_me = SimDet["GiccioneParams"].get("deg", 4)
        self.discretization = SimDet.get("Mechanics Discretization", "P2P1")
        self.discretization_technique = SimDet.get("Technique Discretization", 1)
        self.use_springbc = bool(SimDet.get("springbc", False))

        # Mechanics mesh: LV only.
        self.Mesh = lv_mechanics_mesh(self.parameters, SimDet)
        self.mesh_me = self.Mesh.mesh
        self.facetboundaries_me = self.Mesh.facetboundaries
        self.edgeboundaries_me = self.Mesh.edgeboundaries
        self.matid_me = self.Mesh.matid
        self.ds_me = self.Mesh.ds
        self.dx_me = self.Mesh.dx

        self.f0_me = self.Mesh.f0
        self.s0_me = self.Mesh.s0
        self.n0_me = self.Mesh.n0

        # Cavity and perfusion pressures are mutable Expressions used by the driver.
        self.LVCavityvol = Expression(("vol"), vol=0.0, degree=2)
        self.RVCavityvol = Expression(("vol"), vol=0.0, degree=2)
        self.LVCavitypres = Expression(("pres"), pres=0.0, degree=2)
        self.RVCavitypres = Expression(("pres"), pres=0.0, degree=2)
        self.p_a = Expression(("coronary"), coronary=0.0, degree=2)
        self.p_v = Expression(("venous"), venous=1300.0, degree=2)

        # LV endocardial area is used by Forms.LVV0constrainedE and retained for compatibility.
        self.LVendo_area_me = Expression(("val"), val=0.0, degree=2)
        self.LVendo_area_me.val = assemble(
            Constant(1.0) * self._surface_measure(SimDet["LVendoid"]),
            form_compiler_parameters={"representation": "uflacs"},
        )

        self._build_function_spaces()
        self.Ftotal, self.Jac, self.bcs = self.Problem()

    def default_parameters(self):
        return {"probeloc": [3.5, 0.0, -2.0]}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _surface_measure(self, marker):
        if isinstance(marker, list):
            m = None
            for mid in marker:
                term = self.ds_me(mid, domain=self.mesh_me, subdomain_data=self.facetboundaries_me)
                m = term if m is None else m + term
            return m
        return self.ds_me(marker, domain=self.mesh_me, subdomain_data=self.facetboundaries_me)

    def _build_function_spaces(self):
        if self.discretization == "P1P1":
            Velem = VectorElement("CG", self.mesh_me.ufl_cell(), 1, quad_scheme="default")
        else:
            Velem = VectorElement("CG", self.mesh_me.ufl_cell(), 2, quad_scheme="default")

        # In this minimal poroelastic formulation, subspace 1 is pore/fluid pressure.
        Qelem = FiniteElement("CG", self.mesh_me.ufl_cell(), 1, quad_scheme="default")
        Qelem._quad_scheme = "default"
        # Original LV branch behavior:
        # - springbc=False: add 5 Real Lagrange multipliers to remove rigid-body modes.
        # - springbc=True : do not add the Real block; the epicardial spring-dashpot support
        #                   replaces the rigid-body constraints, matching the original code path.
        if self.use_springbc:
            self.W = FunctionSpace(self.mesh_me, MixedElement([Velem, Qelem]))
        else:
            Relem = FiniteElement("Real", self.mesh_me.ufl_cell(), 0, quad_scheme="default")
            Relem._quad_scheme = "default"
            VRelem = MixedElement([Relem, Relem, Relem, Relem, Relem])
            self.W = FunctionSpace(self.mesh_me, MixedElement([Velem, Qelem, VRelem]))

        Quadelem = FiniteElement(
            "Quadrature", self.mesh_me.ufl_cell(), degree=self.deg_me, quad_scheme="default"
        )
        Quadelem._quad_scheme = "default"
        Telem2 = TensorElement(
            "Quadrature", self.mesh_me.ufl_cell(), degree=self.deg_me,
            shape=2 * (3,), quad_scheme="default"
        )
        Telem2._quad_scheme = "default"
        for e in Telem2.sub_elements():
            e._quad_scheme = "default"

        self.Quad = FunctionSpace(self.mesh_me, Quadelem)
        self.TF = FunctionSpace(self.mesh_me, Telem2)
        self.Q = FunctionSpace(self.mesh_me, "CG", 1)
        self.QDG = FunctionSpace(self.mesh_me, "DG", 0)
        self.V_CG1 = VectorFunctionSpace(self.mesh_me, "CG", 1)

        self.we_n = Function(self.W.sub(0).collapse())
        self.w_me = Function(self.W)
        self.w_me_n = Function(self.W)
        self.dw_me = TrialFunction(self.W)
        self.wtest_me = TestFunction(self.W)

    def _current(self):
        if self.use_springbc:
            u, p = self.w_me.split(deepcopy=True)
            return u, p, None
        u, p, c = self.w_me.split(deepcopy=True)
        return u, p, c

    def _previous(self):
        if self.use_springbc:
            u, p = self.w_me_n.split(deepcopy=True)
            return u, p, None
        u, p, c = self.w_me_n.split(deepcopy=True)
        return u, p, c

    def _poro_params(self):
        return {
            "permeability": self.SimDet.get("permeability", Constant(0.0)),
            "beta_a": self.SimDet.get("beta_a", Constant(0.0)),
            "beta_v": self.SimDet.get("beta_v", Constant(0.0)),
        }

    # ------------------------------------------------------------------
    # Boundary conditions and variational problem
    # ------------------------------------------------------------------
    def set_BCs(self):
        W = self.W
        facetboundaries = self.facetboundaries_me
        bcs = []

        # Original behavior:
        # - without springbc, fix z-displacement on topid and use Real multipliers for remaining modes.
        # - with springbc, do not add this top z DirichletBC; the epicardial spring support stabilizes
        #   the model, while optional fix_surf constraints can still be supplied explicitly.
        if not self.use_springbc:
            topid = self.SimDet["topid"]
            bcs.append(
                DirichletBC(W.sub(0).sub(2), Expression(("0.0"), degree=2), facetboundaries, topid)
            )

        # Optional additional fixed surface(s), retained for compatibility with existing SimDet.
        if "fix_surf" in self.SimDet:
            fix_surf = self.SimDet["fix_surf"]
            markers = fix_surf if isinstance(fix_surf, list) else [fix_surf]
            for marker in markers:
                bcs.append(
                    DirichletBC(
                        W.sub(0), Expression(("0.0", "0.0", "0.0"), degree=2),
                        facetboundaries, marker,
                    )
                )
        return bcs

    def _springbc_residual(self, u_me, u_me_n, v_me, N_me, ds_me):
        """Original LV epicardial spring/dashpot boundary support.

        Normal and tangential spring-dashpot terms are applied on the epicardial marker.
        This mirrors the original LV branch in MEmodel3.py.  The optional moving-base
        compensation is also retained through Forms.topspringbc() when spring_atbase=True.
        """
        dim = u_me.ufl_shape[0]
        I = Identity(dim)
        epiid = self.SimDet["epiid"]
        topid = self.SimDet["topid"]

        k_spring = self.SimDet.get("springparam", [2.0e3, 2.0e3])
        c_damping = self.SimDet.get("dashpotparam", [2.0e2, 2.0e1])
        epiid_Kadj_coeff = self.SimDet.get("epiid_Kadj_coeff", [10.0, 10.0])

        # Original LV normal component used a spatially varying multiplier self.Mesh.poissonF
        # read from the HDF5 dataset "varyingspring" when available; otherwise it is zero in
        # the original mesh helper.  To avoid silently disabling the normal spring if the dataset
        # is absent, allow a fallback value through spring_poisson_fallback.
        poissonF = getattr(self.Mesh, "poissonF", Constant(1.0))
        if self.SimDet.get("spring_poisson_fallback", False):
            poissonF = Constant(float(self.SimDet.get("spring_poisson_fallback", 1.0)))

        u_rate = u_me - u_me_n
        Pn = outer(N_me, N_me)
        Pt = I - Pn

        F3_epi = (
            inner(
                Pn * (k_spring[0] * epiid_Kadj_coeff[0] * poissonF * u_me + c_damping[0] * u_rate),
                v_me,
            ) * ds_me(epiid)
            + inner(
                Pt * (k_spring[1] * epiid_Kadj_coeff[1] * u_me + c_damping[1] * u_rate),
                v_me,
            ) * ds_me(epiid)
        )

        if self.SimDet.get("spring_atbase", False):
            a_, b_ = self.GetTopSpring()
            F3_epi -= a_ * inner(b_, v_me) * ds_me(topid)

        return F3_epi

    def Problem(self):
        GuccioneParams = self.SimDet["GiccioneParams"]
        mesh_me = self.mesh_me
        dx_me = self.dx_me
        ds_me = self.ds_me
        W_me = self.W
        w_me = self.w_me
        w_me_n = self.w_me_n
        dw_me = self.dw_me
        N_me = FacetNormal(mesh_me)
        X_me = SpatialCoordinate(mesh_me)

        if self.use_springbc:
            du, dp = TrialFunctions(W_me)
            u_me, p_me = split(w_me)
            u_me_n, p_me_n = split(w_me_n)
            v_me, q_me = TestFunctions(W_me)
            c_me = None
        else:
            du, dp, dc = TrialFunctions(W_me)
            u_me, p_me, c_me = split(w_me)
            u_me_n, p_me_n, c_me_n = split(w_me_n)
            v_me, q_me, cq = TestFunctions(W_me)

        bcs_elas = self.set_BCs()
        LVendoid = self.SimDet["LVendoid"]
        LVPid = self.SimDet.get("LVPid", LVendoid)
        RVendoid = self.SimDet.get("RVendoid", 1000)
        RVPid = self.SimDet.get("RVPid", RVendoid)

        params = {
            "mesh": mesh_me,
            "facetboundaries": self.facetboundaries_me,
            "facet_normal": N_me,
            "mixedfunctionspace": W_me,
            "mixedfunction": w_me,
            "displacement_variable": u_me,
            "pressure_variable": p_me,
            "lv_volconst_variable": [],
            "lv_constrained_vol": self.LVCavityvol,
            "rv_volconst_variable": [],
            "rv_constrained_vol": self.RVCavityvol,
            "LVendoid": LVendoid,
            "RVendoid": RVendoid,
            "epiid": self.SimDet["epiid"],
            "topid": self.SimDet["topid"],
            "aortaid": None,
            "LVPid": LVPid,
            "RVPid": RVPid,
            "aortic_vplane": None,
            "mitral_vplane": None,
            "first_rv_valve": None,
            "second_rv_valve": None,
            "septumid": None,
            "aorta_wall": None,
            "pulm_wall": None,
            "apxid": self.SimDet.get("apxid", None),
            "aorta_int_wall": None,
            "aorta_ext_wall": None,
            "aorta_ring": None,
            "LVendo_comp": 2,
            "RVendo_comp": 1000,
            "fiber": self.f0_me,
            "sheet": self.s0_me,
            "sheet-normal": self.n0_me,
            "growth_tensor": None,
            "material model": GuccioneParams["Passive model"],
            "material params": GuccioneParams["Passive params"],
            "incompressible": GuccioneParams["incompressible"],
            "LVendo_area": self.LVendo_area_me,
            "lv_constrained_pres": self.LVCavitypres,
            "rv_constrained_pres": self.RVCavitypres,
        }
        uflforms = Forms(params)
        self.uflforms = uflforms

        self.t_a = Function(self.Quad)
        self.t_a.vector()[:] = 0.0
        activeparams = {
            "mesh": mesh_me,
            "dx": dx_me,
            "deg": GuccioneParams["deg"],
            "facetboundaries": self.facetboundaries_me,
            "facet_normal": N_me,
            "displacement_variable": u_me,
            "pressure_variable": p_me,
            "fiber": self.f0_me,
            "sheet": self.s0_me,
            "sheet-normal": self.n0_me,
            "t_a": self.t_a,
            "Threshold_Potential": 0.9,
            "growth_tensor": None,
            "material model": GuccioneParams.get("Active model", {"Name": "Time-varying"}),
            "HomogenousActivation": GuccioneParams.get("HomogenousActivation", True),
        }
        if "Active params" in GuccioneParams:
            activeparams["material params"] = GuccioneParams["Active params"]
        self.activeforms = activeForms(activeparams)

        Fmat = uflforms.Fmat()
        Sactive = self.activeforms.PK2StressTensor()
        state_obj = self.parameters["state_obj"]
        pp = self._poro_params()
        permeability = pp["permeability"]
        beta_a = pp["beta_a"]
        beta_v = pp["beta_v"]

        # Poroelastic mass/Darcy balance.
        source = (
            beta_a * (self.p_a - p_me) * q_me * dx_me
            - beta_v * (p_me - self.p_v) * q_me * dx_me
        )
        F_mass = (
            -(1.0 / state_obj.dt.dt) * inner((u_me - u_me_n), grad(q_me)) * dx_me
            - dot(-permeability * grad(p_me), grad(q_me)) * dx_me
            - source
        )

        # Poroelastic momentum balance: passive skeleton + volumetric skeleton stress - fluid pressure.
        F_mom = (
            self.LVCavitypres * inner(v_me, N_me) * self._surface_measure(LVPid)
            + inner(grad(v_me), uflforms.poro_PK_1()) * dx_me
            - inner(grad(v_me), p_me * Identity(u_me.ufl_shape[0])) * dx_me
        )

        # Active stress. Keep active_region hook for future coupling to spatial models.
        if "active_region" in self.SimDet and self.SimDet["active_region"]:
            F_active = Constant(0.0) * q_me * dx_me
            for regionid in self.SimDet["active_region"]:
                F_active += inner(Fmat * Sactive, grad(v_me)) * dx_me(int(regionid))
        else:
            F_active = inner(Fmat * Sactive, grad(v_me)) * dx_me

        if self.use_springbc:
            F_support = self._springbc_residual(u_me, u_me_n, v_me, N_me, ds_me)
        else:
            # Rigid-body constraints retained from the original LV pressure-control branch.
            Wrigid = (
                inner(as_vector([c_me[0], c_me[1], 0.0]), u_me)
                + inner(as_vector([0.0, 0.0, c_me[2]]), cross(X_me, u_me))
                + inner(as_vector([c_me[3], 0.0, 0.0]), cross(X_me, u_me))
                + inner(as_vector([0.0, c_me[4], 0.0]), cross(X_me, u_me))
            )
            F_support = derivative(Wrigid, w_me, self.wtest_me) * dx_me

        Ftotal = F_mass + F_mom + F_active + F_support
        Jac = derivative(Ftotal, w_me, dw_me)
        return Ftotal, Jac, bcs_elas

    # ------------------------------------------------------------------
    # Solver and state management
    # ------------------------------------------------------------------
    def Solver(self):
        solverparams = {
            "Jacobian": self.Jac,
            "F": self.Ftotal,
            "w": self.w_me,
            "boundary_conditions": self.bcs,
            "Type": self.SimDet.get("Type", 0),
            "mesh": self.mesh_me,
            "mode": 1,
        }
        if "abs_tol" in self.SimDet:
            solverparams["abs_tol"] = self.SimDet["abs_tol"]
        if "rel_tol" in self.SimDet:
            solverparams["rel_tol"] = self.SimDet["rel_tol"]
        return NSolver(solverparams)

    def UpdateVar(self):
        self.w_me_n.assign(self.w_me)

    def Reset(self):
        self.w_me.assign(self.w_me_n)

    # ------------------------------------------------------------------
    # Accessors used by the closed-loop driver
    # ------------------------------------------------------------------
    def GetDisplacement(self):
        u, p, c = self._current()
        u.rename("u_", "u_")
        return u

    def GetP(self):
        u, p, c = self._current()
        p.rename("p_", "p_")
        return p

    def GetPorePressure(self):
        return self.GetP()

    def GetPorePressure2(self):
        p = self.GetP()
        vol = assemble(Constant(1.0) * self.dx_me)
        if abs(vol) < 1e-30:
            return 0.0
        return assemble(p * self.dx_me) / vol

    def GetFmat(self):
        return self.uflforms.Fmat()

    def GetFiberstrain(self, F_ref):
        return self.uflforms.fiberstrain(F_ref=F_ref)

    def GetFiberstrainUL(self):
        F_Identity = Identity(self.GetDisplacement().ufl_domain().geometric_dimension())
        return self.uflforms.fiberstrain(F_ref=F_Identity)

    def GetIMP(self):
        return self.uflforms.IMP()

    def GetIMP2(self):
        return self.uflforms.IMP2()

    def Getfstress(self):
        return self.uflforms.fiberstress() + self.activeforms.fiberstress()

    def GetLVP(self):
        return self.LVCavitypres.pres

    def GetLVV(self):
        return self.LV_closedsurf()

    def LV_closedsurf(self):
        if self.use_springbc:
            return self.uflforms.LVcavityvol_mvb()
        return self.uflforms.LVcavityvol()

    def GetTopSpring(self):
        return self.uflforms.topspringbc()

    def GetRVP(self):
        return 0.0

    def GetRVV(self):
        return 0.0

    def GetWallVolume(self):
        return assemble(self.uflforms.J() * self.dx_me)

    def GetDarcy(self):
        p = self.GetP()
        return -self._poro_params()["permeability"] * grad(p)

    def GetFlowRate(self):
        p = self.GetP()
        pp = self._poro_params()
        return assemble(pp["beta_a"] * (self.p_a - p) * self.dx_me)

    def GetFlowRate_n(self):
        u_n, p_n, c_n = self._previous()
        pp = self._poro_params()
        return assemble(pp["beta_v"] * (p_n - self.p_v) * self.dx_me)

    def GetFlowRateEndo(self):
        J = self.uflforms.J()
        area = assemble(Constant(1.0) * self._surface_measure(self.SimDet["LVendoid"]))
        if abs(area) < 1e-30:
            return 0.0
        return assemble((J - 1.0) * self._surface_measure(self.SimDet["LVendoid"])) / area

    def GetMass(self):
        m = project(self.uflforms.J() - 1.0, self.W.sub(1).collapse())
        m.rename("m_", "m_")
        return m

    def GetMassGrad(self):
        mg = project(grad(self.uflforms.J() - 1.0), self.W.sub(0).collapse())
        mg.rename("grad_m_", "grad_m_")
        return mg

    def GetNormaldirtctionDarcy(self):
        V0 = FunctionSpace(self.mesh_me, "CG", 1)
        flux = project(div(self.GetDarcy()), V0)
        flux.rename("NormalDarcyFlux_", "NormalDarcyFlux_")
        return flux

    def GetNormaldirtction(self):
        mesh_me = self.mesh_me
        V0 = FunctionSpace(mesh_me, "CG", 1)
        T = Function(V0, name="T_transmural")
        u = TrialFunction(V0)
        v = TestFunction(V0)
        a = inner(grad(u), grad(v)) * self.dx_me
        L = Constant(0.0) * v * self.dx_me
        bc_endo = DirichletBC(V0, Constant(0.0), self.facetboundaries_me, self.SimDet["LVendoid"])
        bc_epi = DirichletBC(V0, Constant(1.0), self.facetboundaries_me, self.SimDet["epiid"])
        solve(a == L, T, [bc_endo, bc_epi], solver_parameters={"linear_solver": "mumps"})
        Wv = VectorFunctionSpace(mesh_me, "CG", 1)
        gT = project(grad(T), Wv)
        gT.rename("gradT", "gradT")
        eps = Constant(1e-24)
        n_tm_expr = gT / sqrt(inner(gT, gT) + eps)
        return T, n_tm_expr

    def GetSActive(self):
        Sactive = self.activeforms.PK2StressTensor()
        i, j = indices(2)
        Sactive_ = project(self.f0_me[i] * Sactive[i, j] * self.f0_me[j], self.QDG)
        Sactive_.rename("Sact", "Sact")
        return Sactive_

    # ------------------------------------------------------------------
    # Kept as explicit unsupported hooks for old postprocessing paths.
    # ------------------------------------------------------------------
    def GetFiberNaturalStrain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.GetFmat()
        return self.activeforms.CalculateFiberNaturalStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def GetFiberBiotStrain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.GetFmat()
        return self.activeforms.CalculateFiberBiotStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def GetFiberGreenStrain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.GetFmat()
        return self.activeforms.CalculateFiberGreenStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def GetDeformedBasis(self, params):
        raise NotImplementedError(
            "GetDeformedBasis/fiber regeneration was removed from the minimal LV poroelastic model. "
            "Use the original MEmodel3.py if you need unloading/fiber regeneration."
        )
